from pynamodb.attributes import UnicodeAttribute
from pynamodb.models import Model


class Conversation(Model):
    class Meta:
        table_name = "conrad-conversation"
        region = "us-east-1"
        # host = "http://localhost:8000" # noqa # TODO: for self-host dynamodb

    SessionId = UnicodeAttribute(hash_key=True)


if not Conversation.exists():
    Conversation.create_table(wait=True, read_capacity_units=5, write_capacity_units=5)
