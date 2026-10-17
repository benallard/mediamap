import json
import threading
from http.client import HTTPConnection
from unittest.mock import patch

from mediamap import books, web

from _media_test_helpers import MediaTestCase


class BookTests(MediaTestCase):
    def test_isbn_normalization_validates_isbn_10_and_13(self):
        self.assertEqual(books.normalize_isbn("0-306-40615-2"), "0306406152")
        self.assertEqual(books.normalize_isbn("978-0-306-40615-7"), "9780306406157")
        with self.assertRaisesRegex(ValueError, "valid ISBN"):
            books.normalize_isbn("9780306406158")

    def test_books_web_search_and_add_by_isbn(self):
        handler = type(
            "TestBooksHandler", (web.MediaMapHandler,),
            {"spec": books.BookSpec(), "log_message": lambda *args: None},
        )
        server = web.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port)
        metadata = {
            "sources": ["google"], "source_ids": {"google": "volume-1"},
            "title": "Example Book", "authors": ["A. Writer"],
            "published_date": "2020-01-01", "isbn_13": "9780306406157",
            "isbn_10": "0306406152", "publisher": "Example Press",
            "description": "A test book.", "subjects": ["Fiction"], "language": "en",
        }
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            page = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertIn("Book shelf", page)
            self.assertIn("Scan barcode", page)
            self.assertIn("html5-qrcode@2.3.8", page)

            with patch.object(web, "lookup_metadata", return_value=metadata) as lookup:
                connection.request(
                    "POST", "/api/search",
                    body=json.dumps({"isbn": "978-0-306-40615-7", "title": "", "author": ""}),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                candidate = json.loads(response.read())[0]
                self.assertEqual(response.status, 200)
                self.assertEqual(candidate["source_id"], "volume-1")
                self.assertEqual(lookup.call_args.kwargs["isbn"], "9780306406157")

            connection.request(
                "POST", "/api/albums", body=json.dumps({"record": candidate}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 201)
            book_id = json.loads(response.read())["id"]

            connection.request("GET", "/api/albums")
            response = connection.getresponse()
            stored = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(stored[0]["id"], book_id)
            self.assertEqual(stored[0]["isbn"], "9780306406157")
            self.assertEqual(stored[0]["authors"], "A. Writer")

            connection.request(
                "POST", "/api/search", body=json.dumps({"isbn": "123", "title": "", "author": ""}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
        finally:
            connection.close()
            server.shutdown()
            thread.join()
            server.server_close()
