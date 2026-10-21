"""Generate a WEBSVC_BASIC_CREDS entry.

Usage:
    python -m ampzip.webservice.hashpw alice@example.com
        Prompts for a password and prints alice@example.com:pbkdf2_sha256$...

    python -m ampzip.webservice.hashpw alice@example.com --gen
        Generates and prints a random password and an email:hash entry.

Set WEBSVC_BASIC_CREDS to the email:hash entries, separated by commas:
    WEBSVC_BASIC_CREDS='alice@x.com:pbkdf2_sha256$...,bob@y.com:pbkdf2_sha256$...'
"""

from __future__ import annotations

import argparse
import getpass
import secrets
import sys

from .pwhash import hash_password


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ampzip.webservice.hashpw",
                                description="Generate a password hash for HTTP Basic authentication.")
    p.add_argument("email", help="Email address used as the HTTP Basic username.")
    p.add_argument("--gen", action="store_true",
                   help="Generate a random password.")
    args = p.parse_args(argv)

    if args.gen:
        secret = secrets.token_urlsafe(24)  # ~32 chars, ~144 bits
        print(f"# Password for {args.email}:")
        print(f"#   {secret}")
    else:
        secret = getpass.getpass(f"Password for {args.email}: ")
        confirm = getpass.getpass("Confirm password: ")
        if secret != confirm:
            print("Passwords do not match.", file=sys.stderr)
            return 1
        if len(secret) < 12:
            print("Warning: the password is shorter than 12 characters. Use --gen to generate one.",
                  file=sys.stderr)

    print(f"{args.email}:{hash_password(secret)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
