#!/usr/bin/env -S uv run

import argparse
import sys
from pathlib import Path
from typing import Callable

from pytubefix import Search

from data_models import SearchResult, SearchEntry


def search_youtube(query: str, max_results: int = 5, filter: Callable[[SearchEntry], bool] = None) -> SearchResult:
    s = Search(query)
    results = []
    search_results = []
    while len(results) < max_results:
        if not search_results:
            s.get_next_results()
            search_results = s.results
        yt = search_results.pop(0)
        if yt.watch_url in [r.url for r in results]:
            continue
        result = SearchEntry(
            url=yt.watch_url,
            title=yt.title,
            length=yt.length,
            publish_date=str(yt.publish_date),
        )
        if filter is None or filter(result):
            results.append(result)
    return SearchResult(count=len(results), results=results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search YouTube for videos matching a query.")
    parser.add_argument("query", type=str, nargs="+", help="Search query")
    parser.add_argument("--max-results", type=int, default=50, help="Maximum number of results to return")
    parser.add_argument("--output-format", type=str, choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--save-file", type=str, help="File to save results")
    parser.add_argument("--min-length", type=int, help="Minimum video length in seconds")
    parser.add_argument("--max-length", type=int, help="Maximum video length in seconds")
    args = parser.parse_args()

    def filter_func(entry: SearchEntry) -> bool:
        if args.min_length is not None and entry.length < args.min_length:
            return False
        if args.max_length is not None and entry.length > args.max_length:
            return False
        return True

    args.query = " ".join(args.query)
    save_file = Path(args.save_file) if args.save_file else None
    if save_file:
        save_file.parent.mkdir(exist_ok=True)
        out_stream = save_file.open("w", encoding="utf-8")
    else:
        out_stream = sys.stdout

    result = search_youtube(args.query, max_results=args.max_results, filter=filter_func)
    match args.output_format:
        case "json":
            output = result.model_dump_json(indent=2)
        case "text":
            output_lines = [f"Found {result.count} results for query: '{args.query}'"]
            for i, entry in enumerate(result.results, 1):
                minutes, seconds = divmod(entry.length, 60)
                output_lines.append(f"{i}. {entry.title} - {entry.url} ({minutes}m{seconds}s)")
            output = "\n".join(output_lines)
        case _:
            raise ValueError(f"Unsupported output format: {args.output_format}")

    print(output, file=out_stream)
    out_stream.close()
