from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    "都市の天気を返す"
    return "快晴"


@tool
def get_templature(city: str) -> str:
    "都市の気温を返す"
    return "摂氏25度"


tool_repository = dict()
tool_repository[get_weather.name] = get_weather
tool_repository[get_templature.name] = get_templature
