specialCharacters = [",", ".", "(", ")", ":", '"', ";", "!","?","@","/","\\"]
count = 0
def preprocess(text):
    text = deleteUrls(text) # Based on the training data, I think that the urls are irrelevant and they should have no impact on the prediction of the model
    for char in specialCharacters:
        text = text.replace(char, "") # Eliminate special characters
    text = text.strip().split(" ") # Split words into array
    print(count)
    return text

def deleteUrls(text):
    global count
    if(haveUrl(text)):
        urlsArray = findUrls(text)
        count = count + len(urlsArray)
        for url in urlsArray:
            text = text.replace(url, "")
    return text

def haveUrl(text):
    if text.find('http://') != -1 or text.find('https://') != -1:
        return True
    else:
        return False

def findUrls(text):
    urls = []
    start1 = 0 # Position start searching for 'http'
    start2 = 0 # Position start searching for 'https'
    while True:
        http = True # If find 'http' string, htpt = True
        pos = text.find('http://', start1)
        if(pos == -1):
            pos = text.find('https://', start2)
            http = False 
        if(pos != -1):
            temp = "" # temp holding value of an url
            for char in range(pos, len(text)):
                if text[char] == " " or text[char] == '\n':
                    break
                temp = temp + text[char]
            urls.append(temp)
            if(http):
                start1 = pos + len(temp)
            else:
                start2 = pos + len(temp)
        else:
            break
    return urls

