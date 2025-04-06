import scrapy

class NewsSpider(scrapy.Spider):
    name = "news"
    start_urls = ["https://kyivindependent.com/tag/ukraine/"]

    def parse(self, response):
        article_links = response.css('a.archiveCard__link::attr(href)').getall()

        for link in article_links:
            if link.startswith("/"):
                yield response.follow(link, callback=self.parse_article)

    def parse_article(self, response):
        title = response.css('h1::text').get()
        paragraphs = response.css('article p::text').getall()

        yield {
            'url': response.url,
            'title': title.strip() if title else '',
            'content': ' '.join(p.strip() for p in paragraphs if p.strip())
        }