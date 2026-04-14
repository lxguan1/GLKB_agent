"""List available GLKB tools."""
import asyncio
from tools import glkb_tools, pubmed_tools

def list_tools():
    all_tools = glkb_tools + pubmed_tools
    print(f"\n=== Available Tools ({len(all_tools)}) ===")
    for tool in all_tools:
        name = getattr(tool, 'name', getattr(tool, '_name', str(tool)))
        desc = getattr(tool, 'description', '')[:100] if hasattr(tool, 'description') else ''
        print(f"  - {name}")
        if desc:
            print(f"    {desc}")
    return all_tools

if __name__ == "__main__":
    list_tools()
