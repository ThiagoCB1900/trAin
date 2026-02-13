from pathlib import Path
import re

content = Path('literature/svr/index.html').read_text(encoding='utf-8')

# Count all opening and closing tags
opening_a = len(re.findall(r'<a\s+[^>]*>', content))
closing_a = len(re.findall(r'</a>', content))

opening_i = len(re.findall(r'<i>', content))
closing_i = len(re.findall(r'</i>', content))

opening_b = len(re.findall(r'<b>', content))
closing_b = len(re.findall(r'</b>', content))

print(f"<a> tags: {opening_a} opening, {closing_a} closing")
print(f"<i> tags: {opening_i} opening, {closing_i} closing")
print(f"<b> tags: {opening_b} opening, {closing_b} closing")

if opening_a != closing_a:
    print(f"\n❌ PROBLEMA: {opening_a - closing_a} tags <a> não fechadas!")
if opening_i != closing_i:
    print(f"\n❌ PROBLEMA: {opening_i - closing_i} tags <i> não fechadas!")
if opening_b != closing_b:
    print(f"\n❌ PROBLEMA: {opening_b - closing_b} tags <b> não fechadas!")

# Find line where problem occurs
if opening_a != closing_a or opening_i != closing_i or opening_b != closing_b:
    lines = content.split('\n')
    count_a = 0
    count_i = 0
    count_b = 0
    
    for i, line in enumerate(lines, 1):
        count_a += len(re.findall(r'<a\s+[^>]*>', line))
        count_a -= len(re.findall(r'</a>', line))
        
        count_i += len(re.findall(r'<i>', line))
        count_i -= len(re.findall(r'</i>', line))
        
        count_b += len(re.findall(r'<b>', line))
        count_b -= len(re.findall(r'</b>', line))
        
        if i > 1150 and (count_a != 0 or count_i != 0 or count_b != 0):
            print(f"\nLinha {i}: <a>={count_a}, <i>={count_i}, <b>={count_b}")
            print(f"  {line[:100]}")
