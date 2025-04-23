import re
from typing import Optional

def parse_time(time_str: Optional[str]) -> int:
    """Converts time string (1h30m15s) to total seconds"""
    if not time_str or time_str.lower() == "unknown":
        return 0
        
    hours = re.search(r'(\d+)h', time_str)
    minutes = re.search(r'(\d+)m', time_str)
    seconds = re.search(r'(\d+)s', time_str)
    
    total = 0
    if hours: total += int(hours.group(1)) * 3600
    if minutes: total += int(minutes.group(1)) * 60
    if seconds: total += int(seconds.group(1))
    
    return total

def format_time(total_seconds: int, style: str = "compact") -> str:
    """Formats seconds into time string with style options
    Args:
        total_seconds: Time duration in seconds
        style: "compact" (1h30m) or "extended" (01:30:00)
    """
    if total_seconds <= 0:
        return "0s" if style == "compact" else "00:00:00"
    
    hours = total_seconds // 3600
    remaining = total_seconds % 3600
    minutes = remaining // 60
    seconds = remaining % 60
    
    if style == "compact":
        parts = []
        if hours > 0: parts.append(f"{hours}h")
        if minutes > 0: parts.append(f"{minutes}m")
        if seconds > 0 or not parts: parts.append(f"{seconds}s")
        return "".join(parts)
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
           