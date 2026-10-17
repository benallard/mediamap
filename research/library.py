"""A fake but recognisable library: 100 LPs and 100 books, with ground-truth families.

Titles, years and genre tags are written from memory (not fetched from MusicBrainz or
Open Library), so treat tags as plausible rather than authoritative. What matters for the
research is that the *structure* is realistic: tags nest in families, some tags belong to
several families (jazz fusion, synth-pop, dystopia, ...), and a few creators appear twice.

CDs are not modelled: they mirror the LPs 1:1.

The tag embedder here is a stand-in for the sentence model: it builds a vector per tag from
its position in the taxonomy (super-family > family > tag), so "post-punk" is near "new wave"
and far from "bebop", and cross-listed tags sit between their families. Swap in the real one
with `minilm_tag_vectors` when running on your machine.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- taxonomies
# family -> tags. A tag listed under several families is a cross-over genre; the first family
# listed is its "primary" one (used for ground-truth labels).
MUSIC_FAMILIES = {
    "rock": ["rock", "classic rock", "hard rock", "blues rock", "psychedelic rock", "progressive rock",
             "glam rock", "soft rock", "pop rock", "folk rock", "country rock", "art rock",
             "krautrock", "jazz fusion"],
    "punk & new wave": ["punk rock", "post-punk", "new wave", "gothic rock", "no wave", "synth-pop", "ska"],
    "alternative": ["alternative rock", "indie rock", "shoegaze", "dream pop", "grunge", "britpop",
                    "noise rock", "art rock", "art pop"],
    "metal": ["heavy metal", "thrash metal", "doom metal", "progressive metal", "hard rock"],
    "jazz": ["jazz", "cool jazz", "modal jazz", "bebop", "hard bop", "free jazz", "jazz fusion",
             "soul jazz", "bossa nova"],
    "blues-soul-funk": ["blues", "delta blues", "soul", "funk", "rhythm and blues", "motown",
                        "gospel", "disco", "soul jazz"],
    "hip hop": ["hip hop", "gangsta rap", "conscious hip hop", "boom bap", "trip hop"],
    "electronic": ["electronic", "ambient", "techno", "house", "idm", "downtempo", "electro",
                   "krautrock", "synth-pop", "trip hop", "disco", "dub", "minimalism"],
    "pop": ["pop", "dance-pop", "synth-pop", "art pop", "soft rock", "pop rock", "disco",
            "singer-songwriter"],
    "folk-country": ["folk", "country", "americana", "singer-songwriter", "folk rock", "country rock"],
    "reggae-world": ["reggae", "dub", "ska", "afrobeat", "latin", "world music", "bossa nova"],
    "classical": ["classical", "baroque", "romantic", "contemporary classical", "minimalism"],
}
# Which family a cross-listed tag counts as when we label an item with ground truth.
MUSIC_PRIMARY = {
    "krautrock": "electronic", "jazz fusion": "jazz", "trip hop": "electronic", "dub": "reggae-world",
    "art pop": "pop", "minimalism": "classical", "singer-songwriter": "folk-country",
}
MUSIC_SUPER = {
    "rock": "guitar", "punk & new wave": "guitar", "alternative": "guitar", "metal": "guitar",
    "jazz": "black american", "blues-soul-funk": "black american", "hip hop": "black american",
    "electronic": "electronic & pop", "pop": "electronic & pop",
    "folk-country": "acoustic & traditional", "reggae-world": "acoustic & traditional",
    "classical": "acoustic & traditional",
}

BOOK_FAMILIES = {
    "science fiction": ["science fiction", "space opera", "cyberpunk", "dystopia", "hard science fiction",
                        "post-apocalyptic", "artificial intelligence"],
    "fantasy": ["fantasy", "epic fantasy", "high fantasy", "urban fantasy", "fairy tale",
                "magical realism", "gothic fiction"],
    "crime & thriller": ["crime", "detective fiction", "noir", "thriller", "spy fiction", "mystery"],
    "literary fiction": ["literary fiction", "modernism", "postmodernism", "satire", "bildungsroman",
                         "magical realism", "dystopia", "existentialism", "war fiction"],
    "classics": ["classic literature", "russian literature", "victorian literature", "gothic fiction",
                 "romance"],
    "historical fiction": ["historical fiction", "war fiction"],
    "poetry & drama": ["poetry", "drama"],
    "history": ["history", "world history", "military history", "ancient history"],
    "science": ["popular science", "physics", "cosmology", "biology", "evolution", "mathematics"],
    "philosophy": ["philosophy", "ethics", "existentialism", "political philosophy", "stoicism"],
    "computing": ["computer science", "programming", "software engineering", "artificial intelligence",
                  "algorithms", "mathematics"],
    "biography": ["biography", "autobiography", "memoir"],
    "society": ["economics", "sociology", "politics", "psychology", "political philosophy"],
    "travel & cooking": ["travel", "nature writing", "mountaineering", "cooking"],
}
BOOK_PRIMARY = {
    "magical realism": "literary fiction", "gothic fiction": "classics", "war fiction": "historical fiction",
    "existentialism": "philosophy", "artificial intelligence": "computing",
}
BOOK_SUPER = {
    "science fiction": "fiction", "fantasy": "fiction", "crime & thriller": "fiction",
    "literary fiction": "fiction", "classics": "fiction", "historical fiction": "fiction",
    "poetry & drama": "fiction",
    "history": "humanities", "philosophy": "humanities", "biography": "humanities",
    "society": "humanities",
    "science": "science & tech", "computing": "science & tech",
    "travel & cooking": "lifestyle",
}

# ---------------------------------------------------------------------- items
# (title, creator, year, "tags, most important first")
LPS = [
    # rock
    ("Led Zeppelin IV", "Led Zeppelin", 1971, "hard rock, blues rock"),
    ("The Dark Side of the Moon", "Pink Floyd", 1973, "progressive rock, psychedelic rock"),
    ("Wish You Were Here", "Pink Floyd", 1975, "progressive rock, art rock"),
    ("Abbey Road", "The Beatles", 1969, "pop rock, psychedelic rock"),
    ("Exile on Main St.", "The Rolling Stones", 1972, "blues rock, classic rock"),
    ("Rumours", "Fleetwood Mac", 1977, "soft rock, pop rock"),
    ("Hotel California", "Eagles", 1976, "soft rock, country rock"),
    ("Born to Run", "Bruce Springsteen", 1975, "classic rock"),
    ("Highway 61 Revisited", "Bob Dylan", 1965, "folk rock, blues rock"),
    ("Machine Head", "Deep Purple", 1972, "hard rock, heavy metal"),
    ("The Rise and Fall of Ziggy Stardust", "David Bowie", 1972, "glam rock, art rock"),
    ("Low", "David Bowie", 1977, "art rock, ambient"),
    ("Tubular Bells", "Mike Oldfield", 1973, "progressive rock, ambient"),
    ("Close to the Edge", "Yes", 1972, "progressive rock"),
    ("Marquee Moon", "Television", 1977, "punk rock, new wave"),
    ("Layla and Other Assorted Love Songs", "Derek and the Dominos", 1970, "blues rock, classic rock"),
    # punk / post-punk
    ("Ramones", "Ramones", 1976, "punk rock"),
    ("Never Mind the Bollocks", "Sex Pistols", 1977, "punk rock"),
    ("London Calling", "The Clash", 1979, "punk rock, ska"),
    ("Unknown Pleasures", "Joy Division", 1979, "post-punk, gothic rock"),
    ("Closer", "Joy Division", 1980, "post-punk, gothic rock"),
    ("Entertainment!", "Gang of Four", 1979, "post-punk"),
    ("Pornography", "The Cure", 1982, "gothic rock, post-punk"),
    ("Disintegration", "The Cure", 1989, "gothic rock, alternative rock"),
    ("Remain in Light", "Talking Heads", 1980, "new wave, funk, art rock"),
    ("Parallel Lines", "Blondie", 1978, "new wave, punk rock"),
    ("Dare", "The Human League", 1981, "synth-pop, new wave"),
    ("Violator", "Depeche Mode", 1990, "synth-pop, new wave"),
    ("Power, Corruption & Lies", "New Order", 1983, "new wave, synth-pop, electronic"),
    ("The Specials", "The Specials", 1979, "ska, new wave"),
    # alternative
    ("Doolittle", "Pixies", 1989, "alternative rock, indie rock"),
    ("Daydream Nation", "Sonic Youth", 1988, "noise rock, alternative rock"),
    ("Loveless", "My Bloody Valentine", 1991, "shoegaze, noise rock"),
    ("Nevermind", "Nirvana", 1991, "grunge, alternative rock"),
    ("OK Computer", "Radiohead", 1997, "alternative rock, art rock"),
    ("Kid A", "Radiohead", 2000, "art rock, electronic"),
    ("Definitely Maybe", "Oasis", 1994, "britpop, alternative rock"),
    ("Parklife", "Blur", 1994, "britpop"),
    ("Turn On the Bright Lights", "Interpol", 2002, "indie rock, post-punk"),
    # metal
    ("Paranoid", "Black Sabbath", 1970, "heavy metal, hard rock"),
    ("Master of Puppets", "Metallica", 1986, "thrash metal"),
    ("Reign in Blood", "Slayer", 1986, "thrash metal"),
    ("The Number of the Beast", "Iron Maiden", 1982, "heavy metal"),
    ("Blackwater Park", "Opeth", 2001, "progressive metal"),
    # jazz
    ("Kind of Blue", "Miles Davis", 1959, "modal jazz, cool jazz"),
    ("Bitches Brew", "Miles Davis", 1970, "jazz fusion"),
    ("A Love Supreme", "John Coltrane", 1965, "modal jazz, free jazz"),
    ("Time Out", "The Dave Brubeck Quartet", 1959, "cool jazz"),
    ("Mingus Ah Um", "Charles Mingus", 1959, "hard bop"),
    ("Head Hunters", "Herbie Hancock", 1973, "jazz fusion, funk"),
    ("Blue Train", "John Coltrane", 1958, "hard bop"),
    ("Song for My Father", "Horace Silver", 1965, "hard bop, soul jazz"),
    ("The Köln Concert", "Keith Jarrett", 1975, "jazz"),
    ("Saxophone Colossus", "Sonny Rollins", 1956, "hard bop"),
    ("Brilliant Corners", "Thelonious Monk", 1957, "bebop, hard bop"),
    ("Moanin'", "Art Blakey & The Jazz Messengers", 1958, "hard bop, soul jazz"),
    # blues / soul / funk / disco
    ("Folk Singer", "Muddy Waters", 1964, "delta blues, blues"),
    ("Live at the Regal", "B.B. King", 1965, "blues"),
    ("King of the Delta Blues Singers", "Robert Johnson", 1961, "delta blues"),
    ("What's Going On", "Marvin Gaye", 1971, "soul, motown"),
    ("Songs in the Key of Life", "Stevie Wonder", 1976, "soul, funk"),
    ("Lady Soul", "Aretha Franklin", 1968, "soul, rhythm and blues"),
    ("Stand!", "Sly & the Family Stone", 1969, "funk, soul"),
    ("Maggot Brain", "Funkadelic", 1971, "funk, psychedelic rock"),
    ("Off the Wall", "Michael Jackson", 1979, "disco, funk"),
    ("C'est Chic", "Chic", 1978, "disco, funk"),
    # electronic
    ("Autobahn", "Kraftwerk", 1974, "krautrock, electronic"),
    ("Trans-Europe Express", "Kraftwerk", 1977, "electronic, synth-pop"),
    ("Phaedra", "Tangerine Dream", 1974, "krautrock, ambient"),
    ("Ege Bamyasi", "Can", 1972, "krautrock"),
    ("Selected Ambient Works 85–92", "Aphex Twin", 1992, "idm, ambient, techno"),
    ("Music for Airports", "Brian Eno", 1978, "ambient"),
    ("Dummy", "Portishead", 1994, "trip hop"),
    ("Mezzanine", "Massive Attack", 1998, "trip hop"),
    ("Homework", "Daft Punk", 1997, "house, electronic"),
    ("Discovery", "Daft Punk", 2001, "house, disco"),
    ("Moon Safari", "Air", 1998, "downtempo, synth-pop"),
    ("Untrue", "Burial", 2007, "downtempo, dub"),
    # hip hop
    ("Illmatic", "Nas", 1994, "hip hop, boom bap"),
    ("The Chronic", "Dr. Dre", 1992, "gangsta rap, hip hop"),
    ("3 Feet High and Rising", "De La Soul", 1989, "hip hop, conscious hip hop"),
    ("It Takes a Nation of Millions to Hold Us Back", "Public Enemy", 1988, "hip hop, conscious hip hop"),
    ("The Low End Theory", "A Tribe Called Quest", 1991, "hip hop, jazz"),
    ("To Pimp a Butterfly", "Kendrick Lamar", 2015, "hip hop, conscious hip hop, jazz"),
    ("Enter the Wu-Tang (36 Chambers)", "Wu-Tang Clan", 1993, "hip hop, boom bap"),
    # folk / country
    ("Blue", "Joni Mitchell", 1971, "singer-songwriter, folk"),
    ("The Freewheelin' Bob Dylan", "Bob Dylan", 1963, "folk, singer-songwriter"),
    ("Nebraska", "Bruce Springsteen", 1982, "folk, singer-songwriter"),
    ("Harvest", "Neil Young", 1972, "folk rock, country rock"),
    ("At Folsom Prison", "Johnny Cash", 1968, "country"),
    ("Red Headed Stranger", "Willie Nelson", 1975, "country, americana"),
    ("Yankee Hotel Foxtrot", "Wilco", 2002, "americana, alternative rock"),
    ("Pink Moon", "Nick Drake", 1972, "folk, singer-songwriter"),
    # pop
    ("Thriller", "Michael Jackson", 1982, "pop, dance-pop"),
    ("Purple Rain", "Prince", 1984, "funk, pop"),
    ("Hounds of Love", "Kate Bush", 1985, "art pop, synth-pop"),
    ("Back to Black", "Amy Winehouse", 2006, "soul, pop"),
    # reggae / world
    ("Catch a Fire", "Bob Marley & The Wailers", 1973, "reggae"),
    ("King Tubby Meets Rockers Uptown", "Augustus Pablo", 1976, "dub, reggae"),
    ("Zombie", "Fela Kuti", 1976, "afrobeat, funk"),
    # classical
    ("Glassworks", "Philip Glass", 1982, "minimalism, contemporary classical"),
    ("Goldberg Variations", "Glenn Gould", 1955, "baroque, classical"),
    ("Tabula Rasa", "Arvo Pärt", 1984, "contemporary classical, minimalism"),
]

BOOKS = [
    # science fiction
    ("Dune", "Frank Herbert", 1965, "science fiction, space opera"),
    ("Foundation", "Isaac Asimov", 1951, "science fiction, space opera"),
    ("I, Robot", "Isaac Asimov", 1950, "science fiction, artificial intelligence"),
    ("Neuromancer", "William Gibson", 1984, "cyberpunk, science fiction"),
    ("Snow Crash", "Neal Stephenson", 1992, "cyberpunk, science fiction"),
    ("Do Androids Dream of Electric Sheep?", "Philip K. Dick", 1968, "science fiction, dystopia"),
    ("Nineteen Eighty-Four", "George Orwell", 1949, "dystopia, science fiction"),
    ("Solaris", "Stanisław Lem", 1961, "science fiction, existentialism"),
    ("The Martian", "Andy Weir", 2011, "hard science fiction"),
    ("The Hitchhiker's Guide to the Galaxy", "Douglas Adams", 1979, "science fiction, satire"),
    # fantasy
    ("The Hobbit", "J. R. R. Tolkien", 1937, "fantasy, high fantasy"),
    ("The Fellowship of the Ring", "J. R. R. Tolkien", 1954, "epic fantasy, high fantasy"),
    ("A Game of Thrones", "George R. R. Martin", 1996, "epic fantasy"),
    ("A Wizard of Earthsea", "Ursula K. Le Guin", 1968, "fantasy, high fantasy"),
    ("Mort", "Terry Pratchett", 1987, "fantasy, satire"),
    ("Guards! Guards!", "Terry Pratchett", 1989, "fantasy, satire"),
    ("American Gods", "Neil Gaiman", 2001, "urban fantasy, fantasy"),
    ("The Lion, the Witch and the Wardrobe", "C. S. Lewis", 1950, "fantasy, fairy tale"),
    # crime & thriller
    ("The Big Sleep", "Raymond Chandler", 1939, "noir, detective fiction"),
    ("The Maltese Falcon", "Dashiell Hammett", 1930, "noir, detective fiction"),
    ("Murder on the Orient Express", "Agatha Christie", 1934, "detective fiction, mystery"),
    ("The Hound of the Baskervilles", "Arthur Conan Doyle", 1902, "detective fiction, gothic fiction"),
    ("Tinker Tailor Soldier Spy", "John le Carré", 1974, "spy fiction, thriller"),
    ("The Girl with the Dragon Tattoo", "Stieg Larsson", 2005, "crime, thriller"),
    ("The Name of the Rose", "Umberto Eco", 1980, "historical fiction, mystery"),
    # literary fiction
    ("One Hundred Years of Solitude", "Gabriel García Márquez", 1967, "magical realism, literary fiction"),
    ("The Master and Margarita", "Mikhail Bulgakov", 1967, "magical realism, satire"),
    ("Ulysses", "James Joyce", 1922, "modernism"),
    ("Mrs Dalloway", "Virginia Woolf", 1925, "modernism"),
    ("The Trial", "Franz Kafka", 1925, "modernism, existentialism"),
    ("The Metamorphosis", "Franz Kafka", 1915, "modernism, existentialism"),
    ("The Catcher in the Rye", "J. D. Salinger", 1951, "bildungsroman, literary fiction"),
    ("The Great Gatsby", "F. Scott Fitzgerald", 1925, "literary fiction, modernism"),
    ("Catch-22", "Joseph Heller", 1961, "satire, war fiction"),
    ("Slaughterhouse-Five", "Kurt Vonnegut", 1969, "postmodernism, science fiction, war fiction"),
    ("Infinite Jest", "David Foster Wallace", 1996, "postmodernism"),
    ("Never Let Me Go", "Kazuo Ishiguro", 2005, "literary fiction, dystopia"),
    ("Steppenwolf", "Hermann Hesse", 1927, "modernism, existentialism"),
    # classics
    ("Crime and Punishment", "Fyodor Dostoevsky", 1866, "russian literature, classic literature"),
    ("War and Peace", "Leo Tolstoy", 1869, "russian literature, historical fiction"),
    ("Anna Karenina", "Leo Tolstoy", 1878, "russian literature, classic literature"),
    ("Pride and Prejudice", "Jane Austen", 1813, "classic literature, romance"),
    ("Jane Eyre", "Charlotte Brontë", 1847, "victorian literature, gothic fiction"),
    ("Wuthering Heights", "Emily Brontë", 1847, "gothic fiction, victorian literature"),
    ("Great Expectations", "Charles Dickens", 1861, "victorian literature, bildungsroman"),
    ("Frankenstein", "Mary Shelley", 1818, "gothic fiction, science fiction"),
    ("Dracula", "Bram Stoker", 1897, "gothic fiction, victorian literature"),
    # historical fiction
    ("Wolf Hall", "Hilary Mantel", 2009, "historical fiction"),
    ("The Pillars of the Earth", "Ken Follett", 1989, "historical fiction"),
    ("All Quiet on the Western Front", "Erich Maria Remarque", 1929, "war fiction, historical fiction"),
    ("I, Claudius", "Robert Graves", 1934, "historical fiction"),
    # poetry & drama
    ("Hamlet", "William Shakespeare", 1603, "drama"),
    ("Waiting for Godot", "Samuel Beckett", 1953, "drama, existentialism"),
    ("Leaves of Grass", "Walt Whitman", 1855, "poetry"),
    ("The Waste Land", "T. S. Eliot", 1922, "poetry, modernism"),
    # history
    ("Sapiens", "Yuval Noah Harari", 2011, "world history, history"),
    ("Guns, Germs, and Steel", "Jared Diamond", 1997, "world history, history"),
    ("The Rise and Fall of the Third Reich", "William L. Shirer", 1960, "history, military history"),
    ("The Guns of August", "Barbara Tuchman", 1962, "military history, history"),
    ("SPQR", "Mary Beard", 2015, "ancient history, history"),
    ("A People's History of the United States", "Howard Zinn", 1980, "history"),
    ("Postwar", "Tony Judt", 2005, "history"),
    # science
    ("A Brief History of Time", "Stephen Hawking", 1988, "cosmology, physics"),
    ("Cosmos", "Carl Sagan", 1980, "cosmology, popular science"),
    ("The Selfish Gene", "Richard Dawkins", 1976, "evolution, biology"),
    ("On the Origin of Species", "Charles Darwin", 1859, "evolution, biology"),
    ("The Elegant Universe", "Brian Greene", 1999, "physics, cosmology"),
    ("Fermat's Last Theorem", "Simon Singh", 1997, "mathematics, popular science"),
    ("Silent Spring", "Rachel Carson", 1962, "nature writing, biology"),
    ("The Structure of Scientific Revolutions", "Thomas Kuhn", 1962, "philosophy, popular science"),
    # philosophy
    ("Meditations", "Marcus Aurelius", 180, "stoicism, philosophy"),
    ("Thus Spoke Zarathustra", "Friedrich Nietzsche", 1883, "philosophy, existentialism"),
    ("Beyond Good and Evil", "Friedrich Nietzsche", 1886, "philosophy, ethics"),
    ("The Myth of Sisyphus", "Albert Camus", 1942, "existentialism, philosophy"),
    ("Man's Search for Meaning", "Viktor Frankl", 1946, "existentialism, memoir"),
    ("Justice", "Michael Sandel", 2009, "political philosophy, ethics"),
    ("Leviathan", "Thomas Hobbes", 1651, "political philosophy"),
    # computing
    ("The C Programming Language", "Brian Kernighan & Dennis Ritchie", 1978, "programming"),
    ("The Pragmatic Programmer", "Andrew Hunt & David Thomas", 1999, "software engineering, programming"),
    ("Structure and Interpretation of Computer Programs", "Harold Abelson & Gerald Sussman", 1985,
     "computer science, programming"),
    ("Introduction to Algorithms", "Thomas Cormen et al.", 1990, "algorithms, computer science"),
    ("The Mythical Man-Month", "Frederick Brooks", 1975, "software engineering"),
    ("Clean Code", "Robert C. Martin", 2008, "software engineering, programming"),
    ("Artificial Intelligence: A Modern Approach", "Stuart Russell & Peter Norvig", 1995,
     "artificial intelligence, computer science"),
    ("The Art of Computer Programming", "Donald Knuth", 1968, "algorithms, mathematics"),
    ("Gödel, Escher, Bach", "Douglas Hofstadter", 1979, "mathematics, artificial intelligence, philosophy"),
    # biography
    ("Long Walk to Freedom", "Nelson Mandela", 1994, "autobiography, biography"),
    ("The Diary of a Young Girl", "Anne Frank", 1947, "memoir, history"),
    ("Steve Jobs", "Walter Isaacson", 2011, "biography"),
    ("Educated", "Tara Westover", 2018, "memoir"),
    ("Einstein: His Life and Universe", "Walter Isaacson", 2007, "biography, physics"),
    # society
    ("The Wealth of Nations", "Adam Smith", 1776, "economics"),
    ("Capital in the Twenty-First Century", "Thomas Piketty", 2013, "economics"),
    ("Thinking, Fast and Slow", "Daniel Kahneman", 2011, "psychology"),
    ("The Communist Manifesto", "Karl Marx & Friedrich Engels", 1848, "political philosophy, economics"),
    # travel, nature & cooking
    ("In Patagonia", "Bruce Chatwin", 1977, "travel"),
    ("Into Thin Air", "Jon Krakauer", 1997, "mountaineering, travel"),
    ("A Walk in the Woods", "Bill Bryson", 1998, "travel, nature writing"),
    ("Salt Fat Acid Heat", "Samin Nosrat", 2017, "cooking"),
    ("Jerusalem", "Yotam Ottolenghi & Sami Tamimi", 2012, "cooking"),
]

MEDIA = {
    "lps": dict(items=LPS, families=MUSIC_FAMILIES, supers=MUSIC_SUPER, primary=MUSIC_PRIMARY,
                era_sigma=10.0),
    "books": dict(items=BOOKS, families=BOOK_FAMILIES, supers=BOOK_SUPER, primary=BOOK_PRIMARY,
                  era_sigma=30.0),
}


# -------------------------------------------------------------------- library
@dataclass
class Library:
    medium: str
    titles: list[str]
    creators: list[str]
    years: np.ndarray
    tags: list[list[str]]            # most important first
    families: list[str]              # ground truth: primary family of the first tag
    tag_families: dict[str, list[str]]
    supers: dict[str, str]
    era_sigma: float
    all_tags: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.titles)

    def label(self, i: int) -> str:
        return f"{self.titles[i]} — {self.creators[i]} ({self.years[i]})"


def load_library(medium: str) -> Library:
    spec = MEDIA[medium]
    tag_families: dict[str, list[str]] = {}
    for family, tags in spec["families"].items():
        for tag in tags:
            tag_families.setdefault(tag, []).append(family)

    titles, creators, years, tags, families = [], [], [], [], []
    unknown = set()
    for title, creator, year, tag_string in spec["items"]:
        item_tags = [t.strip() for t in tag_string.split(",")]
        unknown.update(t for t in item_tags if t not in tag_families)
        titles.append(title)
        creators.append(creator)
        years.append(year)
        tags.append(item_tags)
        first = item_tags[0]
        families.append(spec["primary"].get(first) or tag_families.get(first, ["?"])[0])
    if unknown:
        raise KeyError(f"tags missing from the {medium} taxonomy: {sorted(unknown)}")
    return Library(medium, titles, creators, np.array(years), tags, families, tag_families,
                   spec["supers"], spec["era_sigma"], sorted(tag_families))


# ----------------------------------------------------------------- tag vectors
def _unit(name: str, dim: int) -> np.ndarray:
    seed = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=dim)
    return v / np.linalg.norm(v)


def structural_tag_vectors(lib: Library, dim: int = 128, w_super=0.9, w_family=1.0, w_tag=0.8):
    """Stand-in for a sentence model: each tag = super-family + family + its own noise.

    Siblings come out at cosine ~0.7, cousins in the same super-family ~0.3, strangers ~0,
    and a cross-listed tag lands between its families.
    """
    vectors = {}
    for tag, families in lib.tag_families.items():
        family_part = np.mean([_unit("family:" + f, dim) for f in families], axis=0)
        super_part = np.mean([_unit("super:" + lib.supers[f], dim) for f in families], axis=0)
        v = w_super * super_part + w_family * family_part + w_tag * _unit("tag:" + tag, dim)
        vectors[tag] = v / np.linalg.norm(v)
    return vectors


def minilm_tag_vectors(lib: Library):
    """The real thing: embed each tag on its own with the project's sentence model.

    (Untested here: needs the model weights, which this environment cannot download.)
    """
    from mediamap.embedding import encode_texts

    vectors = encode_texts(lib.all_tags)
    return dict(zip(lib.all_tags, np.asarray(vectors)))
