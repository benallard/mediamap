import json
import threading
from http.client import HTTPConnection
from unittest.mock import Mock, patch

from mediamap import books, web

from _media_test_helpers import MediaTestCase


class BookTests(MediaTestCase):
    def test_isbn_normalization_validates_isbn_10_and_13(self):
        self.assertEqual(books.normalize_isbn("0-306-40615-2"), "0306406152")
        self.assertEqual(books.normalize_isbn("978-0-306-40615-7"), "9780306406157")
        with self.assertRaisesRegex(ValueError, "valid ISBN"):
            books.normalize_isbn("9780306406158")

    def test_book_metadata_merge_stays_with_books_domain(self):
        google = {
            "source": "google", "source_id": "volume-1", "title": "Example",
            "description": "A detailed description.", "subjects": ["Fiction"],
        }
        openlibrary = {
            "source": "openlibrary", "source_id": "/works/OL1W", "title": "Example",
            "subjects": ["Adventure"],
        }
        google_provider = Mock(name="google")
        google_provider.name = "google"
        google_provider.search.return_value = [google]
        openlibrary_provider = Mock(name="openlibrary")
        openlibrary_provider.name = "openlibrary"
        openlibrary_provider.search.return_value = [openlibrary]
        with patch.object(books, "BOOK_PROVIDERS", (google_provider, openlibrary_provider)):
            merged = books.lookup_metadata(title="Example")

        self.assertEqual(merged["sources"], ["google", "openlibrary"])
        self.assertEqual(merged["source_ids"], {
            "google": "volume-1", "openlibrary": "/works/OL1W",
        })
        self.assertEqual(merged["subjects"], ["Fiction", "Adventure"])
        self.assertEqual(merged["description"], "A detailed description.")

    def test_book_refresh_uses_isbn_and_updates_metadata_and_semantic_text(self):
        col = self.open_collection(books.BookSpec())
        col.db.execute(
            """INSERT INTO books(title,authors,isbn,source,source_id,description,
               subjects,semantic_text) VALUES(?,?,?,?,?,?,?,?)""",
            ("Old title", "A. Writer", "9780000000001", "openlibrary",
             "/works/OL1W", "", "[]", "stale text"),
        )
        col.db.commit()
        refreshed = {
            "source_id": "google-volume", "title": "Updated title",
            "authors": ["A. Writer"], "published_date": "2024-06-01",
            "isbn_13": "9780000000001", "description": "New description.",
            "subjects": ["Fiction"], "language": "en",
        }

        with patch.object(books.GOOGLE_BOOKS_PROVIDER, "search", return_value=[refreshed]) as search:
            col.refresh()

        query = search.call_args.args[0]
        self.assertEqual(query.identifier, "9780000000001")
        row = col.db.execute("SELECT * FROM books").fetchone()
        self.assertEqual(row["title"], "Updated title")
        self.assertEqual(row["year"], "2024")
        self.assertEqual(row["description"], "New description.")
        self.assertEqual(json.loads(row["subjects"]), ["Fiction"])
        self.assertIn("Book description: New description.", row["semantic_text"])
        self.assertEqual(row["source"], "openlibrary")
        self.assertEqual(row["source_id"], "/works/OL1W")

    def test_book_refresh_falls_back_to_title_and_author_without_isbn(self):
        spec = books.BookSpec()
        row = {
            "isbn": "", "title": "Example", "authors": "A. Writer",
        }
        with patch.object(books.GOOGLE_BOOKS_PROVIDER, "search", return_value=[]) as search:
            self.assertIsNone(spec.refresh_row(row))

        query = search.call_args.args[0]
        self.assertEqual((query.title, query.creator, query.identifier),
                         ("Example", "A. Writer", None))

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
