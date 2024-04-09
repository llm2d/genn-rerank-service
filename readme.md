# request

```json
{
    "query": "水果",
    "inputs": [
        {
            "id": "1",
            "text": "苹果"
        },
                {
            "id": "2",
            "text": "番茄"
        },
         {
            "id": "3",
            "text": "土豆"
        },
        {
            "id": "4",
            "text": "蔡徐坤"
        }
    ]
}
```

response

```json
{
    "code": 0,
    "message": "重排成功",
    "data": [
        {
            "id": "1",
            "text": "苹果",
            "score": 0.5152262462271403
        },
        {
            "id": "2",
            "text": "番茄",
            "score": 0.5059926834569244
        },
        {
            "id": "3",
            "text": "土豆",
            "score": 0.44221167430412855
        },
        {
            "id": "4",
            "text": "蔡徐坤",
            "score": 0.3636002817386903
        }
    ]
}
```
