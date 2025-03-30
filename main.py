from fastapi import FastAPI
from reactpy import component, html, run
from reactpy.backend.fastapi import configure

#Comando para inicar el servidor: uvicorn main:app --reload
#Explicacion: dentro de la carpeta AIAgents, el archivo main.py tiene la variable app que es la que se ejecuta al iniciar el servidor.
#Modo debug para ver los errores mejor: $env:REACTPY_DEBUG_MODE="1" 
# y luego uvicorn main:app --reload

app = FastAPI()

@component
def TaskList():
    tasks = [
        {"id": 0, "text": "Tarea 1", "priority": 1},
        {"id": 1, "text": "Tarea 2", "priority": 2},
        {"id": 2, "text": "Tarea 3", "priority": 3},
        {"id": 3, "text": "Tarea 4", "priority": 4},
    ]

    lis = [html.li({
        "key": task['id'],
        "style": {
            "color": "black",
            "background": "yellow",
        }
        }, task['text']) for task in tasks] 
    return html.ul(lis)

@component
def App():
    return html.main(
        html.h1("Mis tareas"),
        html.div(TaskList())
    )



@component
def GoodComponent():
    return html.p("This is a good component")

@component
def BadComponent():
    msg = 'this is an error component'
    raise RuntimeError(msg)

@component
def App2(): #prueb. cambiar por App para volver a usar este modulo, o ejecutar con App2 en vez de App en el comando de uvicorn y configure
    return html.section(
        #GoodComponent(), #descomentar para probar
        #BadComponent()
        html.h1("Lista de tareas"),
        html.ul( #lista de elementos
            html.li("Tarea 1"),
            html.li("Tarea 2"),
            html.li("Tarea 3"),
        ),
        html.img({
            "src": "https://t.resfu.com/img_data/shields_basket/55.png?v=5",
            "style": {
                "width": "50%",
            },
            "alt": "Viva er betih", #texto alternativo en caso de que no se cargue la imagen (comentando el src)
        })
    )

configure(app, App)