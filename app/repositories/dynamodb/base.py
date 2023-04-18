import json  # will use orjson in the future
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union

from boto3.dynamodb.types import Binary
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pynamodb.expressions.condition import Condition
from pynamodb.models import Model

ModelType = TypeVar("ModelType", bound=Model)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: Type[ModelType]):
        """
        CRUD object with default methods to Create, Read, Update, Delete (CRUD).

        **Parameters**

        * `model`: A PynamoDB model class
        * `schema`: A Pydantic model (schema) class
        """
        self.model = model

    def query(
        self,
        partition_key: Union[
            str,
            int,
            Binary,
        ],
        sort_key_condition: Optional[Condition] = None,
        filter_condition: Optional[Condition] = None,
    ) -> list[Optional[ModelType]]:
        if sort_key_condition is not None:
            self.model.get()
            return [
                item
                for item in self.model.query(
                    hash_key=partition_key,
                    range_key_condition=sort_key_condition,
                    filter_condition=filter_condition,
                )
            ]
        return [
            item
            for item in self.model.query(
                hash_key=partition_key, filter_condition=filter_condition
            )
        ]

    def get(
        self,
        partition_key: Union[
            str,
            int,
            Binary,
        ],
        sort_key: Optional[
            Union[
                str,
                int,
                Binary,
            ]
        ] = None,
    ) -> Optional[ModelType]:
        return self.model.get(hash_key=partition_key, range_key=sort_key)

    def create(self, *, obj_in: CreateSchemaType) -> ModelType:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)
        db_obj.save()
        db_obj.refresh()
        return db_obj

    def update(
        self, *, db_obj: ModelType, obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        """
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        It internally uses PutItem. The entire object is re-written.
        Any modifications done to the same user by other processes will be lost.
        To avoid this, use Pynamodb.models.update() to preform more fine grained updates.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        obj_data = json.loads(db_obj.to_json())
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        for field in obj_data:
            if field in update_data:
                setattr(obj_data, field, update_data[field])
        db_obj = self.model(**obj_data)
        db_obj.save()
        db_obj.refresh()
        return db_obj

    def remove(
        self,
        db_obj: ModelType,
        partition_key: Union[
            str,
            int,
            Binary,
        ],
        sort_key: Optional[
            Union[
                str,
                int,
                Binary,
            ]
        ],
    ) -> None:
        if sort_key is not None:
            db_obj.delete(
                (
                    (db_obj._hash_keyname == partition_key)
                    & (db_obj._range_keyname == sort_key)
                )
            )
        if sort_key is None:
            db_obj.delete((db_obj._hash_keyname == partition_key))
