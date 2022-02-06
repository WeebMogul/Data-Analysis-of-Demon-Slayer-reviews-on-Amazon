import random
from requests_html import HTMLSession
import time
import json

class CSMReviews:
    def __init__(self, asin):
        self.asin = asin
        self.session = HTMLSession()
        self.agent = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36'}
        self.url = f'https://www.amazon.com/product-reviews/{self.asin}/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=helpful&pageNumber='

    def pagination(self,page):
        
        r = self.session.get(self.url + str(page),headers=self.agent)
        r.html.render(sleep=1)

        book_name = r.html.find('h1 span',first=True).text
        
        if not r.html.find('div[data-hook=review]',first=True):
            return book_name,False
        else:
            return book_name,r.html.find('div[data-hook=review]')
    
    def parse(self,reviews,book_name):

        total = []
        for review in reviews:

            if review.find('a[data-hook=review-title]',first=True) is None:
                title = None
            else:
                title = review.find('a[data-hook=review-title]',first=True).text
            
            if review.find('i[data-hook=review-star-rating] span',first=True) is None:
                rating = None
            else :
                rating = review.find('i[data-hook=review-star-rating] span',first=True).text
           #  review_text = review.find('span[data-hook=review-body] span',first=True).text

            if review.find('span[data-hook=review-body] span',first=True) is None:
                review_text = None
            else : 
                review_text = review.find('span[data-hook=review-body] span',first=True).text.replace("\n"," ").strip()

            data = {'title':title, 'rating' : rating, 'review_text':review_text,'book name':book_name}

            total.append(data)
        
        # print(len(total))
        return total

    def save(self, results,book_name):

        review_total = []
        for volume_review in results:
            for user_review in volume_review:
                review_total.append(user_review)

        with open(book_name + '-reviews.json','w',encoding='utf-8') as f:
            json.dump(review_total,f,ensure_ascii=False)
        


csm_list = ['B07CTBBGZF',
            'B07FR3JRYV',
            'B07JH71GKL',
            'B07LD34FNG',
            'B07NXW6RHP',
            'B07RB7CR6R',
            'B07WFR7S18',
            'B07Z6NJZH8',
            'B082VQSTP6',
            'B084Z2FCGD',
            'B087F9LQXR',
            'B088WRGDQL',
            'B08BGC3276',
            'B08DFSCFFT',
            'B08G7BGPD2',
            'B08JTV5N5J',
            'B08LMN83Y4',
            'B08NT2ZCLZ',
            'B08TGP9BYB',
            'B08ZXDN9CL',
            'B0956GBQP5',
            'B099WFVWKT']

results = []

for volume in csm_list:

    print('Extracting reviews from : '+ volume + '\n')

    csm = CSMReviews(volume)

    for i in range(1,100):
        
        print(f"Getting page {i}")
        book_name,reviews = csm.pagination(i)
        
        time.sleep(random.randint(1,5))
        
        if reviews is not False:
            review_part = csm.parse(reviews,book_name=book_name)
            # review_part = [review.update({'book name':book_name}) for review in review_part]
            results.append(review_part)
        else:
            print("No more pages")
            break
    time.sleep(random.randint(10,15))

csm.save(results,'Demon_Slayer')
print(results)
    
# print(results)