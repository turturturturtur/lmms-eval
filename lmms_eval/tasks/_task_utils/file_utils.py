import os
import re


# XML 1.0 illegal control characters (excludes tab \x09, LF \x0a, CR \x0d)
_ILLEGAL_XML_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


def sanitize_for_excel(val):
    """Remove XML control characters that openpyxl does not support."""
    if isinstance(val, str):
        return _ILLEGAL_XML_CHARS_RE.sub("", val)
    return val


def generate_submission_file(file_name, args, subpath="submissions"):
    if args is None or args.output_path is None:
        # If no output path is specified, use current directory
        path = subpath
    else:
        path = os.path.join(args.output_path, subpath)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, file_name)
    return os.path.abspath(path)
