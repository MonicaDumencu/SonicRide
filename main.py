import logging


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Test log line")


if __name__ == "__main__":
    main()
    