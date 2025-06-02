from therapy import Therapist
from screen import ChampionDetector

def main():
    champion = ChampionDetector("config.json").get_champion_from_file("champ_select.webp")
    print(f"Champion is {champion}")
    detector = Therapist(champion=champion, config_path="config.json")
    detector.run()

if __name__ == "__main__":
    main()