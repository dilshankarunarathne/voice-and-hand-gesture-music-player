songs_dict = {}


def add_song_to_dict(title: str, index: int):
    print("Adding song: " + title + " at index " + str(index))
    songs_dict[title.lower()] = index - 1


def search_song(search_term):
    search_term = search_term.lower().replace('&', '')

    search_term_words = search_term.split(" ")
    search_term_words = [word for word in search_term_words if word not in ["the", "song", "play"]]

    print(songs_dict)

    for song in songs_dict.keys():
        song_normalized = song.lower().replace('&', '')
        print(song_normalized + " <> " + str(search_term_words))
        song_words = song_normalized.split(" ")
        song_words = [word for word in song_words if word != "the"]

        if any(word in song_words for word in search_term_words):
            return songs_dict[song]

    return None
