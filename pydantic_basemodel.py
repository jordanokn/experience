"""
Это типа BaseModel от pydantic? Да ну насмерть
"""

class BaseModel:
   
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls) 
        args_list = list(args)

        for index, (name, annotation) in enumerate(cls.__annotations__.items()):
            if len(args_list) > index:
                value = args_list[index]
            else:
                value = kwargs[name]

            try:
                cls.__validate(name=name, value=value, annotation=annotation)
            except ValueError as e:
                raise e
            else:
                setattr(instance, name, value)
        
        return instance
            
    @classmethod         
    def __validate(cls, name, annotation, value):
        if not isinstance(value, eval(annotation.__name__)):
            real_type = type(value)
            raise ValueError(f"{name} must be an {annotation}. Got {real_type}")
    

class User(BaseModel):
    name: str
    age: int
    skills: list


if __name__ == "__main__":
    """clients code"""
    
    user = User("george", 10, skills=["python"]) 

    user2 = User("kalim", 'sdaf', skills=("sdfsdfdsf", )) # validation error
