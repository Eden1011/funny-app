from therapy import Therapist

def main():
    detector = Therapist("config.json")
    detector.run()

if __name__ == "__main__":
    main()