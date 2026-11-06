def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")
    
    return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))

s1 = "karolin"
s2 = "kathrin"

dist = hamming_distance(s1, s2)
print(f"Hamming Distance between '{s1}' and '{s2}': {dist}")