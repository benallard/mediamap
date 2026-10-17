from unittest.mock import patch

from mediamap import books, cli, films, music

from _media_test_helpers import MediaTestCase


class CliTests(MediaTestCase):
    def test_cli_routes_each_collection_to_its_spec(self):
        with patch.object(cli, "run_command") as run_command:
            cli.main(["books", "add"])
            cli.main(["lps", "list"])
            cli.main(["films", "add"])

        (book_spec, book_args), (lp_spec, lp_args), (film_spec, film_args) = [
            c.args for c in run_command.call_args_list
        ]
        self.assertIsInstance(book_spec, books.BookSpec)
        self.assertEqual(book_args.cmd, "add")
        self.assertIsInstance(lp_spec, music.MusicSpec)
        self.assertEqual(lp_spec.media_type, "lps")
        self.assertEqual(lp_args.cmd, "list")
        self.assertIsInstance(film_spec, films.FilmSpec)
        self.assertEqual(film_args.cmd, "add")

    def test_books_web_cli_uses_https_certificate_arguments(self):
        with patch.object(cli, "serve") as serve:
            cli.main(["books", "web", "--cert", "cert.pem", "--key", "key.pem"])

        spec, host, port, cert, key = serve.call_args.args
        self.assertIsInstance(spec, books.BookSpec)
        self.assertEqual((host, port, cert, key), ("0.0.0.0", 8765, "cert.pem", "key.pem"))
