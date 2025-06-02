import webbrowser
import urllib.parse

def open_youtube_guide(champion_name):
    if not champion_name:
        print("No champion name provided!")
        return False

    search_query = f"{champion_name} pro league guide"
    encoded_query = urllib.parse.quote_plus(search_query)
    youtube_url = f"https://www.youtube.com/results?search_query={encoded_query}"

    try:
        webbrowser.open(youtube_url)
        return True
    except Exception as e:
        print(f"Error opening browser: {e}")
        return False


def open_guide_for_champion(champion_name):
    return open_youtube_guide(champion_name)


if __name__ == "__main__":
    test_champion = "Jinx"
    open_guide_for_champion(test_champion)