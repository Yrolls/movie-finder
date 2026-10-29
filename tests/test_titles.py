from movie_finder.schemas import Movie
from movie_finder.titles import TitleIndex, norm_title, split_year


def M(mid, title, year):
    return Movie(movie_id=mid, title=title, year=year, origin="American", genre="", director="",
                 cast="", wiki_url="", plot="p")


def test_split_year():
    assert split_year("Annabelle (2014)") == ("Annabelle", 2014)
    assert split_year("Annabelle 2014") == ("Annabelle", 2014)
    assert split_year("Annabelle") == ("Annabelle", None)
    assert split_year("2012") == ("2012", None)                   # a title that is a year


def test_title_index_lookup():
    idx = TitleIndex([M("m00001", "Annabelle", 2014), M("m00002", "Mad Max: Fury Road", 2015),
                      M("m00003", "Mad Max: Fury Road", 2015), M("m00004", "It", 1990),
                      M("m00005", "It", 2017)])
    assert [m.movie_id for m in idx.find("annabelle")] == ["m00001"]
    assert [m.movie_id for m in idx.find("Annabele")] == ["m00001"]          # small typo
    assert [m.movie_id for m in idx.find("mad max fury road")] == ["m00002"]  # duplicate rows -> one
    assert [m.movie_id for m in idx.find("It", 2017)] == ["m00005"]
    assert [m.movie_id for m in idx.find("It", 2016)] == ["m00005"]          # off by one year
    assert len(idx.find("It")) == 2
    assert idx.find("The Conjuring") == []
    assert norm_title("The Matrix") == "matrix"


def test_check_title_keeps_numbers_that_are_part_of_the_title():
    import json
    from movie_finder.agent.tools import ToolBox
    from tests.fixtures import make_retriever
    r = make_retriever()
    r.movies["m00002"] = r.movies["m00002"].model_copy(update={"title": "Ship of Ice 2049"})
    box = ToolBox(r)
    out = json.loads(box.run("check_title", json.dumps({"titles": ["Ship of Ice 2049"], "description": "ice ship"})))
    assert out["results"][0]["movie_id"] == "m00002"
