import json
import threading
from http.client import HTTPConnection
from unittest.mock import patch

from mediamap import music, web

from _media_test_helpers import MediaTestCase, album_record


class WebTests(MediaTestCase):
    def test_web_search_returns_gateway_error_when_musicbrainz_is_unavailable(self):
        handler = type(
            "TestLPHandler", (web.MediaMapHandler,),
            {"spec": music.MusicSpec("lps"), "log_message": lambda *args: None},
        )
        server = web.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port)
        try:
            with patch.object(
                web, "search_releases",
                side_effect=music.MusicBrainzError("MusicBrainz is temporarily unavailable."),
            ):
                connection.request(
                    "POST", "/api/search", body=json.dumps({"title": "Album", "artist": "Artist"}),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 502)
                self.assertIn("temporarily unavailable", json.loads(response.read())["error"])
        finally:
            connection.close()
            server.shutdown()
            thread.join()
            server.server_close()

    def test_web_add_route_saves_selected_release_and_lists_it(self):
        spec = music.MusicSpec("lps")
        handler = type(
            "TestLPHandler", (web.MediaMapHandler,),
            {"spec": spec, "log_message": lambda *args: None},
        )
        server = web.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port)
        try:
            with patch.object(web, "fetch_genres", return_value=["post-punk"]), \
                    patch.object(web, "fetch_lastfm_tags", return_value=["instrumental"]):
                connection.request(
                    "POST", "/api/albums",
                    body=json.dumps({"record": album_record(), "genres": []}),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 201)
                self.assertEqual(json.loads(response.read())["id"], 1)

            connection.request("GET", "/api/albums")
            response = connection.getresponse()
            albums = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(albums[0]["title"], "Unknown Pleasures")
            self.assertEqual(albums[0]["genres"], ["post-punk"])
            self.assertEqual(albums[0]["tags"], ["instrumental"])
        finally:
            connection.close()
            server.shutdown()
            thread.join()
            server.server_close()
