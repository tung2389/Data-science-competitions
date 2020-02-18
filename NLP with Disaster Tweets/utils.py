def analyzeDataLength(train_data):
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for post in train_data:
        if len(post) <= 50:
            count1 = count1 + 1
        elif 50 < len(post) <= 100:
            count2 = count2 + 1
        elif 100 < len(post) <= 150:
            count3 = count3 + 1
        elif len(post) > 150:
            count4 = count4 + 1
    
    print("Posts whose lengths are smaller than 50: " , count1)
    print("Posts whose lengths are between 50 and 100: " , count2)
    print("Posts whose lengths are between 100 and 150: " , count3)
    print("Posts whose lengths are bigger than 150: " , count4)