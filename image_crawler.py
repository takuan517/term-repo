from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={"root_dir": './road'})
crawler.crawl(keyword='道', max_num=250)
