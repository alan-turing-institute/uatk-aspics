# Developer guide

## Code hygiene

We use automated tools to format the code.

```shell
# Format all Python code
poetry run black aspics *.py

# Format Markdown docs
prettier --write *.md docs/*.md
```

Install [prettier](https://prettier.io) for Markdown.
