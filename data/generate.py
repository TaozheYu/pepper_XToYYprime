def generate_list(threshold):
    result = [50]
    while True:
        next_num = result[-1] + int(result[-1] * 0.20)
        if next_num > threshold:
            break
        result.append(next_num)
    return result

threshold = 1400
output_list = generate_list(threshold)
print(output_list)
print(len(output_list))
