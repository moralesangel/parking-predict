"""Bucle de sondeo del feed en tiempo real.

Pensado para correr como servicio (`python -m parking.ingest.poller`). Un ciclo
que falla no tumba el bucle: se registra y se espera al siguiente, porque un
hueco de 5 min en la serie es mucho más barato que un poller muerto.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import signal
import sys
import threading
import time
from types import FrameType

import httpx

from parking.config import Settings, get_settings
from parking.ingest.realtime import SnapshotError, capture_snapshot

logger = logging.getLogger(__name__)


class StopSignal:
    """Parada ordenada ante SIGINT/SIGTERM.

    Docker manda SIGTERM al parar el contenedor. Sin esto, el proceso moriría a
    media escritura; con esto, termina el ciclo en curso y sale.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self, signum: int, _frame: FrameType | None) -> None:
        logger.info(
            "recibida señal %s; se para tras el ciclo en curso", signal.Signals(signum).name
        )
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def wait(self, seconds: float) -> bool:
        """Espera `seconds` o hasta que llegue la señal. True si hay que parar."""
        return self._event.wait(seconds)

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.request)


def run_once(client: httpx.Client, settings: Settings) -> bool:
    """Un ciclo. Devuelve `True` si se capturó sin error."""
    try:
        capture_snapshot(client, settings)
    except SnapshotError:
        # El feed devolvió algo inesperado. Es un problema de datos, no de red:
        # se registra con traza para poder revisar el esquema.
        logger.exception("snapshot inválido; se omite este ciclo")
        return False
    except httpx.HTTPError:
        logger.exception("fallo de red tras agotar los reintentos; se omite este ciclo")
        return False
    except OSError:
        logger.exception("fallo de escritura; se omite este ciclo")
        return False
    return True


def run(
    settings: Settings | None = None,
    *,
    max_cycles: int | None = None,
    stop: StopSignal | None = None,
    interval_seconds: float | None = None,
) -> int:
    """Sondea hasta recibir señal de parada o completar `max_cycles`.

    `max_cycles` e `interval_seconds` existen para los tests y para ejecuciones
    puntuales; en servicio se dejan en `None`, el bucle es indefinido y la
    cadencia sale de `Settings`, donde tiene un suelo de 60 s que protege la
    API de origen.
    """
    settings = settings or get_settings()
    stop = stop or StopSignal()
    interval = settings.poll_interval_seconds if interval_seconds is None else interval_seconds

    logger.info(
        "poller en marcha: dataset=%s intervalo=%ds destino=%s",
        settings.realtime_dataset,
        interval,
        settings.realtime_raw_dir,
    )

    failures = 0
    cycles = 0
    with httpx.Client(timeout=settings.request_timeout_seconds) as client:
        while not stop.requested and (max_cycles is None or cycles < max_cycles):
            started = time.monotonic()
            if run_once(client, settings):
                failures = 0
            else:
                failures += 1
                logger.warning("ciclos fallidos consecutivos: %d", failures)
            cycles += 1

            if max_cycles is not None and cycles >= max_cycles:
                break
            # Descontar lo que tardó el ciclo para que la cadencia no derive.
            remaining = max(0.0, interval - (time.monotonic() - started))
            if stop.wait(remaining):
                break

    logger.info("poller detenido tras %d ciclos", cycles)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sondea el feed de sensores de aparcamiento.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="captura un solo snapshot y sale (útil para comprobar la ingesta)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        stream=sys.stdout,
    )
    # UTC en los logs: los datos están en UTC y mezclar zonas al depurar un
    # desfase de timestamps es una fuente de errores gratuita.
    logging.Formatter.converter = time.gmtime

    settings = get_settings()
    stop = StopSignal()
    stop.install()

    started = dt.datetime.now(dt.UTC)
    failures = run(settings, max_cycles=1 if args.once else None, stop=stop)
    logger.info("tiempo total: %s", dt.datetime.now(dt.UTC) - started)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
