"""Discord message importer for processing exported chat histories."""

import json
import os
import csv
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from discord_integration.trade_parser import TradeMessageParser, ParsedTrade
from discord_integration.trade_analyzer import TradeHistoryAnalyzer
from discord_integration.db_persistence import DiscordDBPersistence


class DiscordMessageImporter:
    """Import and process Discord message exports from various formats."""
    
    SUPPORTED_FORMATS = ['json', 'csv', 'txt']
    
    def __init__(self):
        self.parser = TradeMessageParser()
        self.analyzer = TradeHistoryAnalyzer()
        self.db = DiscordDBPersistence()
        self.import_dir = Path("discord_exports")
        self.import_dir.mkdir(exist_ok=True)
    
    def import_from_file(self, filepath: str) -> Dict[str, Any]:
        """Import messages from an exported file."""
        path = Path(filepath)
        
        if not path.exists():
            return {"success": False, "error": f"File not found: {filepath}"}
        
        ext = path.suffix.lower().lstrip('.')
        
        if ext == 'json':
            return self._import_json(path)
        elif ext == 'csv':
            return self._import_csv(path)
        elif ext == 'txt':
            return self._import_txt(path)
        else:
            return {"success": False, "error": f"Unsupported format: {ext}"}
    
    def _import_json(self, path: Path) -> Dict[str, Any]:
        """Import from DiscordChatExporter JSON format."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            channel_info = {
                "id": data.get("channel", {}).get("id", "unknown"),
                "name": data.get("channel", {}).get("name", path.stem),
                "guild_id": data.get("guild", {}).get("id", "unknown"),
                "guild_name": data.get("guild", {}).get("name", "Unknown Server")
            }
            
            messages = []
            for msg in data.get("messages", []):
                messages.append({
                    "id": msg.get("id"),
                    "author": {"username": msg.get("author", {}).get("name", "Unknown")},
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("timestamp"),
                    "embeds": msg.get("embeds", [])
                })
            
            return self._process_messages(messages, channel_info)
            
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _import_csv(self, path: Path) -> Dict[str, Any]:
        """Import from CSV format (DiscordChatExporter CSV)."""
        try:
            messages = []
            channel_name = path.stem
            
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    messages.append({
                        "id": row.get("ID", row.get("id", str(len(messages)))),
                        "author": {"username": row.get("Author", row.get("author", "Unknown"))},
                        "content": row.get("Content", row.get("content", "")),
                        "timestamp": row.get("Date", row.get("timestamp", "")),
                        "embeds": []
                    })
            
            channel_info = {
                "id": "csv_import",
                "name": channel_name,
                "guild_id": "honey_drip",
                "guild_name": "Honey Drip Network"
            }
            
            return self._process_messages(messages, channel_info)
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _import_txt(self, path: Path) -> Dict[str, Any]:
        """Import from plain text format (copy-pasted messages)."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.strip().split('\n')
            messages = []
            current_msg = {"content": "", "author": "Unknown", "timestamp": None}
            msg_id = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_msg["content"]:
                        messages.append({
                            "id": str(msg_id),
                            "author": {"username": current_msg["author"]},
                            "content": current_msg["content"],
                            "timestamp": current_msg["timestamp"] or datetime.now().isoformat(),
                            "embeds": []
                        })
                        msg_id += 1
                        current_msg = {"content": "", "author": "Unknown", "timestamp": None}
                    continue
                
                if " — " in line and ("Today at" in line or "Yesterday at" in line or "/" in line):
                    parts = line.split(" — ", 1)
                    if len(parts) == 2:
                        if current_msg["content"]:
                            messages.append({
                                "id": str(msg_id),
                                "author": {"username": current_msg["author"]},
                                "content": current_msg["content"],
                                "timestamp": current_msg["timestamp"] or datetime.now().isoformat(),
                                "embeds": []
                            })
                            msg_id += 1
                        
                        current_msg = {
                            "author": parts[0].strip(),
                            "timestamp": parts[1].strip(),
                            "content": ""
                        }
                        continue
                
                if current_msg["content"]:
                    current_msg["content"] += " " + line
                else:
                    current_msg["content"] = line
            
            if current_msg["content"]:
                messages.append({
                    "id": str(msg_id),
                    "author": {"username": current_msg["author"]},
                    "content": current_msg["content"],
                    "timestamp": current_msg["timestamp"] or datetime.now().isoformat(),
                    "embeds": []
                })
            
            channel_info = {
                "id": "txt_import",
                "name": path.stem,
                "guild_id": "honey_drip",
                "guild_name": "Honey Drip Network"
            }
            
            return self._process_messages(messages, channel_info)
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _process_messages(self, messages: List[Dict], channel_info: Dict) -> Dict[str, Any]:
        """Process imported messages through the trade parser and analyzer."""
        print(f"\nProcessing {len(messages)} messages from {channel_info['name']}...")
        
        saved = 0
        
        for msg in messages:
            embeds = msg.get("embeds", [])
            if embeds:
                embed_text = []
                for embed in embeds:
                    if embed.get("title"):
                        embed_text.append(embed["title"])
                    if embed.get("description"):
                        embed_text.append(embed["description"])
                    for field in embed.get("fields", []):
                        embed_text.append(f"{field.get('name', '')}: {field.get('value', '')}")
                if embed_text:
                    msg["content"] = msg.get("content", "") + " " + " | ".join(embed_text)
        
        trades = self.parser.parse_messages(messages)
        print(f"Parsed {len(trades)} trades")
        
        if trades:
            saved = self.db.save_trades(trades, channel_info["id"], channel_info["guild_id"])
            print(f"Saved {saved} new trades to database")
            
            self.db.save_channel_config(
                channel_info["guild_id"],
                channel_info["guild_name"],
                channel_info["id"],
                channel_info["name"]
            )
        
        self.analyzer.clear_trades()
        self.analyzer.add_trades(trades)
        
        profile = self.analyzer.analyze()
        patterns = profile.patterns if profile else []
        strategy = self.analyzer.generate_replication_strategy()
        
        profile_id = None
        if trades and profile:
            for pattern in patterns:
                self.db.save_pattern(pattern, channel_info["id"])
            
            profile_id = self.db.save_trader_profile(
                profile, 
                channel_info["id"], 
                channel_info["guild_id"]
            )
            
            if profile_id:
                self.db.save_replication_strategy(strategy, profile_id)
        
        summary = self.parser.get_trade_summary(trades)
        
        return {
            "success": True,
            "channel": channel_info["name"],
            "messages_processed": len(messages),
            "trades_found": len(trades),
            "trades_saved": saved,
            "patterns": [p.to_dict() for p in patterns] if patterns else [],
            "profile": profile.to_dict() if profile else {},
            "strategy": strategy,
            "summary": summary
        }
    
    def import_all_exports(self) -> Dict[str, Any]:
        """Import all files from the discord_exports directory."""
        results = []
        
        for ext in self.SUPPORTED_FORMATS:
            for filepath in self.import_dir.glob(f"*.{ext}"):
                print(f"\nImporting {filepath.name}...")
                result = self.import_from_file(str(filepath))
                result["filename"] = filepath.name
                results.append(result)
        
        total_trades = sum(r.get("trades_found", 0) for r in results if r.get("success"))
        total_saved = sum(r.get("trades_saved", 0) for r in results if r.get("success"))
        
        return {
            "files_processed": len(results),
            "successful": sum(1 for r in results if r.get("success")),
            "failed": sum(1 for r in results if not r.get("success")),
            "total_trades_found": total_trades,
            "total_trades_saved": total_saved,
            "results": results
        }
    
    def import_from_text(self, text: str, channel_name: str = "manual_import") -> Dict[str, Any]:
        """Import directly from pasted text."""
        temp_file = self.import_dir / f"{channel_name}.txt"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        result = self.import_from_file(str(temp_file))
        
        return result


def run_import():
    """CLI function to run the importer."""
    importer = DiscordMessageImporter()
    
    print("=" * 60)
    print("Discord Message Importer")
    print("=" * 60)
    print(f"\nLooking for exports in: {importer.import_dir.absolute()}")
    print(f"Supported formats: {', '.join(importer.SUPPORTED_FORMATS)}")
    
    results = importer.import_all_exports()
    
    print("\n" + "=" * 60)
    print("Import Summary")
    print("=" * 60)
    print(f"Files processed: {results['files_processed']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Total trades found: {results['total_trades_found']}")
    print(f"Total trades saved: {results['total_trades_saved']}")
    
    for r in results['results']:
        status = "✓" if r.get("success") else "✗"
        print(f"\n{status} {r.get('filename', 'unknown')}")
        if r.get("success"):
            print(f"   Trades: {r.get('trades_found', 0)}")
            if r.get("strategy"):
                strat = r["strategy"]
                print(f"   Bias: {strat.get('directional_bias', 'N/A')}")
                print(f"   Top symbols: {', '.join(strat.get('preferred_symbols', [])[:5])}")
        else:
            print(f"   Error: {r.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    run_import()
