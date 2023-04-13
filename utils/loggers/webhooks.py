import pymsteams
import asyncio 

class Webhook_Teams_Training():
    def __init__(self, desc):
        _url = "https://aivkr.webhook.office.com/webhookb2/fd912cfc-096c-4333-baae-df86b7046061@e6fba773-c87f-46fd-adfb-7bf190cdc27c/IncomingWebhook/dc000ff9f6474dbf9146a4194a8a02de/d9f9d088-106a-4214-9396-174a0baad5c4"

        self.connector = pymsteams.async_connectorcard(_url)
        self.desc = desc
        self.loop = asyncio.get_event_loop()

    def get_connector(self):
        return self.connector
    
    def send_message(self, msg, title=None):
        if title != None:
            assert isinstance(title, str), f"title variable should be str"
            self.connector.title(title)

        if isinstance(msg, str):
            self.connector.text(msg)
        elif isinstance(msg, dict):
            _section = pymsteams.cardsection()
            for key, val in msg.items():
                key, val = str(key), str(val)
                _section.addFact(key, val)
            self.connector.addSection(_section)
        elif isinstance(msg, list):
            for _msg in msg:
                if isinstance(_msg, str):
                    _section = pymsteams.cardsection()
                    _section.text(_msg)
                    self.connector.addSection(_section)
                elif isinstance(_msg, dict):
                    _section = pymsteams.cardsection()
                    for key, val in _msg.items():
                        key, val = str(key), str(val)
                        _section.addFact(key, val)
                    self.connector.addSection(_section)
        
        self.connector.text(self.desc)
        self.loop.run_until_complete(self.connector.send())

    def send_link_button(self, link, msg=None, title=None):
        if title != None:
            assert isinstance(title, str), f"title variable should be str"
            self.connector.title(title)
        
        if msg != None:
            assert isinstance(msg, str), f"msg variable should be str"
            self.connector.text(msg)
        
        assert isinstance(link, str), f"link variable should be str"
        self.connector.addLinkButton(link)

        self.loop.run_until_complete(self.connector.send())

    def change_url(self, url):
        assert isinstance(url, str), f"url variable should be str"
        self.connector.newhookurl(url)


if __name__ == '__main__':
    engine = webhook_teams_training("TEST")
    # engine.send_message(msg={"a": 1}, title="TEST")
    engine.send_message(['a', 'b', 'c', {"dict": 'sdfk'}], "TEST")