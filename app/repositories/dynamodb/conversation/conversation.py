from app.models.dynamodb.conversation import Conversation
from app.repositories.dynamodb.base import CRUDBase
from app.schemas.conversation import (  # noqa
    Conversation,
    ConversationBase,
    ConversationCreate,
)


class CRUDContact(CRUDBase[Conversation, ConversationBase, ConversationCreate]):
    pass


conversation = CRUDContact(Conversation)
