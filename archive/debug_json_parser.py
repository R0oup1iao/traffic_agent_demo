
import re
import json

def parse_json_garbage(args_str):
    print(f"Testing string: {args_str}")
    # Current implementation
    json_objects = re.findall(r'\{[^{}]*\}', args_str)
    print(f"Regex found: {json_objects}")
    
    # Better implementation using brace counting
    results = []
    stack = 0
    start_index = -1
    
    for i, char in enumerate(args_str):
        if char == '{':
            if stack == 0:
                start_index = i
            stack += 1
        elif char == '}':
            stack -= 1
            if stack == 0 and start_index != -1:
                results.append(args_str[start_index:i+1])
                start_index = -1
    
    print(f"Brace counting found: {results}")
    return results

# Test cases
test_1 = '{"destination":"中关村","origin":"北京西站","time":"now"}{"location":"北京西站至中关村沿线区域"}'
test_2 = '{"a": 1, "b": {"c": 2}}' # Nested
test_3 = '{"a": 1} some garbage {"b": 2}'

print("--- Test 1 ---")
parse_json_garbage(test_1)
print("\n--- Test 2 ---")
parse_json_garbage(test_2)
print("\n--- Test 3 ---")
parse_json_garbage(test_3)
