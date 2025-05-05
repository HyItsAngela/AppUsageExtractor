import re
import logging
from fuzzywuzzy import process, fuzz

logger = logging.getLogger(__name__)

def clean_text(text):
    """Basic text cleaning - apply general rules."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Time Parsing/Formatting (Minutes-based, 'Xh Ym' format)
def validate_time_format(time_str, regex_pattern):
    """Validates time string using the specific regex for minutes-based format."""
    if not time_str:
        return False
    return bool(re.fullmatch(regex_pattern, time_str, re.IGNORECASE))

def parse_time(time_str, regex_pattern):
    """Parses time string (e.g., '1h 30m') into total minutes using the notebook's logic."""
    if not time_str:
        return 0
    match = re.fullmatch(regex_pattern, time_str, re.IGNORECASE)
    if not match:
        return 0
    total_minutes = 0
    groups = match.groups()
    try:
        if groups[0] is not None and groups[1] is not None:
            total_minutes += int(groups[0]) * 60 + int(groups[1])
        elif groups[2] is not None:
            total_minutes += int(groups[2]) * 60
        elif groups[3] is not None:
            total_minutes += int(groups[3])
    except (ValueError, IndexError):
        return 0
    logger.debug(f"Parsed time '{time_str}' to {total_minutes} minutes.")
    return total_minutes

def format_minutes(total_minutes):
    """Formats total minutes back into 'Xh Ym' string format."""
    if total_minutes < 0:
        return "Invalid Time"
    if total_minutes == 0:
        return "0m"
    hours = total_minutes // 60
    minutes = total_minutes % 60
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if not parts:
        return "0m"
    return "".join(parts)

def parse_time_hms_to_seconds(time_str_hms):
    """Parse time string in format like '1h30m15s' into total seconds"""
    if not time_str_hms or not isinstance(time_str_hms, str):
        return None  

    total_seconds = 0
    valid = False
    try:
        hours_match = re.search(r'(\d+)h', time_str_hms)
        minutes_match = re.search(r'(\d+)m', time_str_hms)
        seconds_match = re.search(r'(\d+)s', time_str_hms)

        if hours_match:
            total_seconds += int(hours_match.group(1)) * 3600
            valid = True
        if minutes_match:
            total_seconds += int(minutes_match.group(1)) * 60
            valid = True
        if seconds_match:
            total_seconds += int(seconds_match.group(1))
            valid = True

        temp_str = time_str_hms
        if hours_match:
            temp_str = temp_str.replace(hours_match.group(0), '', 1)
        if minutes_match:
            temp_str = temp_str.replace(minutes_match.group(0), '', 1)
        if seconds_match:
            temp_str = temp_str.replace(seconds_match.group(0), '', 1)

        if temp_str:  
            logger.warning(f"Invalid characters remaining in hms string '{time_str_hms}' after parsing: '{temp_str}'")
            return None  

        if not valid: 
            logger.warning(f"No valid h, m, or s parts found in hms string: '{time_str_hms}'")
            return None if time_str_hms else 0

    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing hms time string '{time_str_hms}': {e}")
        return None  

    return total_seconds

# Fuzzy Matching
def get_best_match(query, choices, score_cutoff=85):
    """Simple fuzzy matching function."""
    if not query or not choices:
        return None, 0
    try:
        best_match, score = process.extractOne(query, choices, scorer=fuzz.token_sort_ratio)
        if score >= score_cutoff:
            return best_match, score
        else:
            return None, score
    except Exception as e:
        logger.error(f"Error during original fuzzy matching for '{query}': {e}")
        return None, 0

def validate_and_correct_usage(text):
    """
    Combines:
    1. Custom character replacements (r→m, etc.)
    2. Smart regex-based format correction (5→s, etc.)
    """
    if not text:
        return ""
    OCR_CHAR_REPLACEMENTS = {
        'r': 'm', 'n': 'm', 'I': '1', 'l': '1', 't': '1',
        'i': '1', 'L': '1', 'O': '0', 'o': '0', ' ': '',
    }
    corrected = text
    for error, fix in OCR_CHAR_REPLACEMENTS.items():
        corrected = corrected.replace(error, fix)
    valid_format_hms = re.compile(r"^(\d+h)?(\d+m)?(\d+s)?$")
    if valid_format_hms.fullmatch(corrected) and corrected:
        return corrected
    numbers = re.findall(r"\d+", corrected)
    units = re.findall(r"[hms]", corrected.lower())
    reconstructed = ""
    if not numbers:
        return ""
    if not units:
        if len(numbers) == 1:
            reconstructed = f"{numbers[0]}s"
        else:
            return ""
    else:
        num_idx = 0
        unit_idx = 0
        temp_parts = {'h': None, 'm': None, 's': None}
        while num_idx < len(numbers) and unit_idx < len(units):
            current_num = numbers[num_idx]
            current_unit = units[unit_idx]
            if temp_parts[current_unit] is None:
                temp_parts[current_unit] = current_num
                num_idx += 1
                unit_idx += 1
            else:
                unit_idx += 1
        parts = []
        if temp_parts['h']:
            parts.append(f"{temp_parts['h']}h")
        if temp_parts['m']:
            parts.append(f"{temp_parts['m']}m")
        if temp_parts['s']:
            parts.append(f"{temp_parts['s']}s")
        reconstructed = "".join(parts)
    if valid_format_hms.fullmatch(reconstructed) and reconstructed:
        return reconstructed
    else:
        logger.warning(f"Usage correction failed: Result '{reconstructed}' invalid hms (Original: '{text}')")
        return ""

def match_app_name(ocr_text, app_names_list, threshold=90, secondary_threshold=80):
    """Your original two-tier matching function for app names."""
    if not ocr_text or not app_names_list:
        return None, 0
    valid_app_names = [name for name in app_names_list if isinstance(name, str)]
    if not valid_app_names:
        return None, 0
    try:
        result = process.extractOne(
            ocr_text.lower(),
            [name.lower() for name in valid_app_names],
            scorer=fuzz.token_sort_ratio
        )
    except Exception as e:
        logger.error(f"Error in match_app_name extractOne: {e}")
        return None, 0
    if not result:
        return None, 0
    matched_lower, score = result
    original_name = next((name for name in valid_app_names if name.lower() == matched_lower), None)
    if not original_name:
        return None, 0
    if score >= threshold:
        return original_name, score
    elif score >= secondary_threshold:
        return original_name, score
    else:
        return None, 0

def fuzzy_replace_characters(text, matched_app_name, char_threshold=85, word_threshold=80):
    """Strict character-level replacement."""
    if not matched_app_name or not text:
        return text
    overall_score = fuzz.token_sort_ratio(text.lower(), matched_app_name.lower())
    if overall_score < word_threshold:
        return text
    aligned = []
    i = 0
    j = 0
    len_text = len(text)
    len_matched = len(matched_app_name)
    while i < len_text and j < len_matched:
        text_char = text[i]
        matched_char = matched_app_name[j]
        if text_char.lower() == matched_char.lower():
            aligned.append(text_char)
            i += 1
            j += 1
        else:
            char_score = fuzz.ratio(text_char.lower(), matched_char.lower())
            if char_score >= char_threshold:
                aligned.append(matched_char)
                i += 1
                j += 1
            else:
                aligned.append(text_char)
                i += 1
    while i < len_text:
        aligned.append(text[i])
        i += 1
    return ''.join(aligned)

def enhance_text_correction(text, region_type, app_names_list=None, config=None):
    """
    Enhanced correction function *only* for app names, using multi-tier matching
    and character replacement. For other types, it just returns the input text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    stripped_text = text.strip()
    if region_type == "app_name" and app_names_list:
        name_match_threshold = config.get('fuzzy_name_threshold', 90) if config else 90
        name_match_secondary_threshold = config.get('fuzzy_name_secondary_threshold', 75) if config else 75
        char_replace_char_threshold = config.get('fuzzy_char_threshold', 85) if config else 85
        char_replace_word_threshold = config.get('fuzzy_word_threshold', 80) if config else 80
        matched_name, score = match_app_name(
            stripped_text,
            app_names_list,
            name_match_threshold,
            name_match_secondary_threshold
        )
        if matched_name:
            corrected_name = fuzzy_replace_characters(
                stripped_text,
                matched_name,
                char_replace_char_threshold,
                char_replace_word_threshold
            )
            return corrected_name
        else:
            return stripped_text
    else:
        return stripped_text