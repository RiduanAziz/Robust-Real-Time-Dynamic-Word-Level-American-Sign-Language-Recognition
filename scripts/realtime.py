from __future__ import annotations

import time


def main() -> None:
    print("Realtime inference placeholder.")
    for _ in range(3):
        time.sleep(0.1)
    print("Realtime pipeline ready.")


if __name__ == "__main__":
    main()
