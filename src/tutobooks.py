import json
import os
import random
import shutil
import sys
from pathlib import Path

TIMEOUT = 60 * 60
MAX_LOC = 300


def nb_to_py(nb_path, py_path):
    f = open(nb_path)
    content = f.read()
    f.close()
    nb = json.loads(content)
    py = ''
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            # Is it a shell cell?
            if cell["source"] and cell["source"][0] and cell["source"][0][0] == "!":
                # It's a shell cell
                py += '"""shell\n'
                py += "".join(cell["source"]) + "\n"
                py += '"""\n\n'
            else:
                # It's a Python cell
                py += "".join(cell["source"]) + "\n\n"
        elif cell["cell_type"] == "markdown":
            py += '"""\n'
            py += "".join(cell["source"]) + "\n"
            py += '"""\n\n'
    # Save file
    f = open(py_path, "w")
    f.write(py)
    f.close()
    # Format file with Black
    os.system("black " + py_path)
    # Shorten lines
    py = open(py_path).read()
    try:
        py = _shorten_lines(py)
    finally:
        f = open(py_path, "w")
        f.write(py)
        f.close()


def py_to_nb(py_path, nb_path, fill_outputs=True):
    f = open(py_path)
    py = f.read()
    f.close()
    cells = []
    loc = 0
    while py:
        e, cell_type, py, tag = _get_next_script_element(py)
        lines = e.split("\n")

        if all(line == "" for line in lines):
            continue

        if lines and not lines[0]:
            lines = lines[1:]
        source = [line + "\n" for line in lines]
        # Drop last newline char
        if source and not source[-1].strip():
            source = source[:-1]
        if tag == "shell":
            source = ["!" + line for line in source]
            cell_type = "code"
        if tag == "inline":
            source = ["%" + line for line in source]
            cell_type = "code"
        if tag != "invisible" and source:
            cell = {"cell_type": cell_type, "source": source}
            if cell_type == "code":
                cell["outputs"] = []
                cell["metadata"] = {"colab_type": "code"}
                cell["execution_count"] = 0
                loc += _count_locs(source)
            else:
                cell["metadata"] = {"colab_type": "text"}
            cells.append(cell)
    notebook = {}
    for key in NB_BASE.keys():
        notebook[key] = NB_BASE[key]
    notebook["metadata"]["colab"]["name"] = str(py_path).split("/")[-1][:-3]
    notebook["cells"] = cells
    if loc > MAX_LOC:
        raise ValueError(
            "Found %d lines of code, but expected fewer than %d" % (loc, MAX_LOC)
        )

    f = open(nb_path, "w")
    f.write(json.dumps(notebook, indent=1, sort_keys=True))
    f.close()
    if fill_outputs:
        print("Generating ipynb")
        parent_dir = Path(nb_path).parent
        current_files = os.listdir(parent_dir)
        try:
            os.system(
                "jupyter nbconvert --to notebook --execute --debug "
                + str(nb_path)
                + " --inplace"
                + " --ExecutePreprocessor.timeout="
                + str(TIMEOUT)
            )
        finally:
            new_files = os.listdir(parent_dir)
            for fname in new_files:
                if fname not in current_files:
                    fpath = parent_dir / fname
                    if os.path.isdir(fpath):
                        print("Removing created folder:", fname)
                        shutil.rmtree(fpath)
                    else:
                        print("Removing created file:", fname)
                        os.remove(fpath)




def validate(py):
    """Validate the format of a tutobook script.

    Specifically:
        - validate headers
        - validate style with black
    """
    lines = py.split("\n")
    if not lines[0].startswith('"""'):
        raise ValueError('Missing `"""`-fenced header at top of script.')
    if not lines[1].startswith("Title: "):
        raise ValueError("Missing `Title:` field.")
    if not lines[2].startswith("Author: ") and not lines[2].startswith("Authors: "):
        raise ValueError("Missing `Author:` field.")
    if not lines[3].startswith("Date created: "):
        raise ValueError("Missing `Date created:` field.")
    if not lines[4].startswith("Last modified: "):
        raise ValueError("Missing `Last modified:` field.")
    if not lines[5].startswith("Description: "):
        raise ValueError("Missing `Description:` field.")
    description = lines[5][len("Description: ") :]
    if not description:
        raise ValueError("Missing `Description:` field content.")
    if not description[0] == description[0].upper():
        raise ValueError("Description field content must be capitalized.")
    if not description[-1] == ".":
        raise ValueError("Description field content must end with a period.")
    if len(description) > 100:
        raise ValueError("Description field content must be less than 100 chars.")
    for i, line in enumerate(lines):
        if line.startswith('"""') and line.endswith('"""') and len(line) > 3:
            raise ValueError(
                'Do not use single line `"""`-fenced comments. '
                "Encountered at line %d" % (i,)
            )
    for i, line in enumerate(lines):
        if line.endswith(" "):
            raise ValueError(
                "Found trailing space on line %d; line: `%s`" % (i, line)
            )
    # Validate style with black
    fpath = "/tmp/" + str(random.randint(1e6, 1e7)) + ".py"
    f = open(fpath, "w")
    pre_formatting = "\n".join(lines)
    f.write(pre_formatting)
    f.close()
    os.system("black " + fpath)
    f = open(fpath)
    formatted = f.read()
    f.close()
    os.remove(fpath)
    if formatted != pre_formatting:
        raise ValueError(
            "You python file did not follow `black` conventions. "
            "Run `black your_file.py` to autoformat it."
        )


def _count_locs(lines):
    loc = 0
    string_open = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not string_open:
            if not line.startswith('"""'):
                loc += 1
            else:
                if not line.endswith('"""'):
                    string_open = True
        else:
            if line.startswith('"""'):
                string_open = False
    return loc


def _shorten_lines(py):
    max_len = 90
    lines = []
    for line in py.split("\n"):
        if len(line) <= max_len:
            lines.append(line)
            continue
        i = 0
        while len(line) > max_len:
            line = line.lstrip()
            if " " not in line[1:]:
                lines.append(line)
                break
            else:
                short_line = line[:max_len]
                line = line[max_len:]
                if " " in short_line:
                    reversed_short_line = short_line[::-1]
                    index = reversed_short_line.find(" ") + 1
                    line = short_line[-index:] + line
                    short_line = short_line[:-index]

                lines.append(short_line.lstrip())
            i += 1
            if i > 10:
                raise
        lines.append(line.lstrip())
    return "\n".join(lines)


def _get_next_script_element(py):
    lines = py.split("\n")
    assert lines
    elines = []
    i = 0
    tag = None
    if lines[0].startswith('"""'):
        assert len(lines) >= 2
        etype = "markdown"
        if len(lines[0]) > 3:
            tag = lines[0][3:]
            if tag not in ["shell", "invisible", "inline"]:
                raise ValueError("Found unknown cell tag:", tag)
        lines = lines[1:]
    else:
        etype = "code"

    for i, line in enumerate(lines):
        if line.startswith('"""'):
            break
        else:
            elines.append(line)

    if etype == "markdown":
        py = "\n".join(lines[i + 1 :])
    else:
        py = "\n".join(lines[i:])
    e = "\n".join(elines)

    return e, etype, py, tag


def _parse_header(header):
    lines = header.split("\n")
    title = lines[0][len("Title: ") :]
    author_line = lines[1]
    if author_line.startswith("Authors"):
        author = author_line[len("Authors: ") :]
        auth_field = "Authors"
    else:
        author = author_line[len("Author: ") :]
        auth_field = "Author"
    date_created = lines[2][len("Date created: ") :]
    last_modified = lines[3][len("Last modified: ") :]
    description = lines[4][len("Description: ") :]
    return {
        "title": title,
        "author": author,
        "auth_field": auth_field,
        "date_created": date_created,
        "last_modified": last_modified,
        "description": description,
    }


def _make_output_code_blocks(md):
    lines = md.split("\n")
    output_lines = []
    final_lines = []
    is_inside_backticks = False

    def is_output_line(line, prev_line, output_lines):
        if line.startswith("    ") and len(line) >= 5:
            if output_lines or (lines[i - 1].strip() == "" and line.strip()):
                return True
        return False

    def flush(output_lines, final_lines):
        final_lines.append('<div class="k-default-codeblock">')
        final_lines.append("```")
        if len(output_lines) == 1:
            line = output_lines[0]
            final_lines.append(line[4:])
        else:
            for line in output_lines:
                final_lines.append(line[4:])
        final_lines.append("```")
        final_lines.append("</div>")

    for i, line in enumerate(lines):
        if line.startswith("```"):
            is_inside_backticks = not is_inside_backticks
            final_lines.append(line)
            continue

        if is_inside_backticks:
            final_lines.append(line)
            continue

        if i > 0 and is_output_line(line, lines[-1], output_lines):
            output_lines.append(line)
        elif not line:
            if output_lines:
                if output_lines[-1]:
                    output_lines.append(line)
            else:
                final_lines.append(line)
        else:
            if output_lines:
                flush(output_lines, final_lines)
                output_lines = []
            final_lines.append(line)
    if output_lines:
        flush(output_lines, final_lines)
    return "\n".join(final_lines)


NB_BASE = {
    "metadata": {
        "colab": {
            "collapsed_sections": [],
            "name": "",  # FILL ME
            "private_outputs": False,
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.7.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd not in {"nb2py", "py2nb"}:
        raise ValueError(
            "Specify a command: either "
            "`nb2py source_filename.ipynb target_filename.py` or "
            "`py2nb source_filename.py target_file name.ipynb"
        )
    if len(sys.argv) < 4:
        raise ValueError("Specify a source filename and a target filename")
    source = sys.argv[2]
    target = sys.argv[3]

    if cmd == "py2nb":
        if not source.endswith(".py"):
            raise ValueError(
                "The source filename should be a Python file. Got:", source
            )
        if not target.endswith(".ipynb"):
            raise ValueError(
                "The target filename should be a notebook file. Got:", target
            )
        py_to_nb(source, target)
    if cmd == "nb2py":
        if not source.endswith(".ipynb"):
            raise ValueError(
                "The source filename should be a notebook file. Got:", source
            )
        if not target.endswith(".py"):
            raise ValueError(
                "The target filename should be a Python file. Got:", target
            )
        nb_to_py(source, target)
