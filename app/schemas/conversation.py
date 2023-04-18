from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import UUID4, BaseModel, EmailStr


# Shared properties
class ConversationBase(BaseModel):
    user_id: str
    conversation_id: str
    message: str

    class Config:
        use_enum_values = True
        orm_mode = True


# Properties to receive via API on creation
class ConversationCreate(ConversationBase):
    pass


class ConversationUpdate(ConversationBase):
    pass


class Conversation(ConversationCreate):
    pass


class Conversations(BaseModel):
    pass
