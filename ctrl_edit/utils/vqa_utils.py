OBJECT_EXIST_ABSENT_PROMPT = [
    {
        "Input": "A dog. Not cat.",
        "Output": [
            "Question: Is there a dog present in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is there a cat present in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: What is the animal in the image? Choices: ['dog', 'cat', 'wolf'] Answer: dog",
            "Question: Is the animal in the image a cat? Choices: ['yes', 'no'] Answer: no",
            "Question: Is the animal in the image a dog? Choices: ['yes', 'no'] Answer: yes",
            "Question: How many dogs are there in the image? Choices: ['one', 'two', 'three', 'more than three', 'none'] Answer: one",
            "Question: Is the kite in the image clearly visible? Choices: ['completely visible', 'partially visible', 'barely visible', 'not visible'] Answer: completely visible",
        ],
    },
    {
        "Input": "A bird. Not fish.",
        "Output": [
            "Question: What do you see in the image? Choices: ['bird', 'fish'] Answer: bird",
            "Question: What is in the image? Choices: ['bird', 'fish'] Answer: bird",
            "Question: Is there a fish in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a bird in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: How many birds are there in the image? Choices: ['one', 'two', 'three', 'none'] Answer: one",
            "Question: Is the horse in the image clearly visible? Choices: ['completely visible', 'partially visible', 'barely visible', 'not visible'] Answer: completely visible",
        ],
    },
    {
        "Input": "A horse. Not cow.",
        "Output": [
            "Question: What is the animal in the image? Choices: ['horse', 'cow'] Answer: horse",
            "Question: What do you see in the image? Choices: ['horse', 'cow'] Answer: horse",
            "Question: Is there a cow in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a horse in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: How many horses are there in the image? Choices: ['one', 'two', 'three', 'none'] Answer: one",
            "Question: Is the horse in the image clearly visible? Choices: ['completely visible', 'partially visible', 'barely visible', 'not visible'] Answer: completely visible",
        ],
    },
    {
        "Input": "Two dogs",
        "Output": [
            "Question: What type of animal is in the image? Choices: ['dog', 'cat'] Answer: dog",
            "Question: What do you see in the image? Choices: ['one dog', 'two dogs', 'three dogs', 'none'] Answer: two dogs",
            "Question: Is there a cat in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Are there dogs in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: How many dogs are there in the image? Choices: ['one', 'two', 'three', 'none'] Answer: two",
            "Question: Is the kite in the image clearly visible? Choices: ['completely visible', 'partially visible', 'barely visible', 'not visible'] Answer: completely visible",
        ],
    },
    {
        "Input": "Two sandwiches",
        "Output": [
            "Question: What type of food is in the image? Choices: ['sandwich', 'pizza'] Answer: sandwich",
            "Question: What do you see in the image? Choices: ['one sandwich', 'two sandwiches', 'three sandwiches', 'none'] Answer: two sandwiches",
            "Question: Is there a pizza in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Are there sandwiches in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: How many sandwiches are there in the image? Choices: ['one', 'two', 'three', 'none'] Answer: two",
            "Question: Does the image look deformed? Choices: ['slightly', 'moderately', 'severely'] Answer: slightly",
        ],
    },
    {
        "Input": "A kite. Not balloon.",
        "Output": [
            "Question: What do you see in the image? Choices: ['kite', 'balloon', 'bird'] Answer: kite",
            "Question: What is in the image? Choices: ['kite', 'balloon', 'plane'] Answer: kite",
            "Question: Is there a balloon in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a kite in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is the kite in the image clearly visible? Choices: ['completely visible', 'partially visible', 'barely visible', 'not visible'] Answer: completely visible",
        ],
    },
    {
        "Input": "A bottle of wine. Not beer.",
        "Output": [
            "Question: What type of drink is in the image? Choices: ['wine', 'beer', 'whiskey', 'vodka'] Answer: wine",
            "Question: Is the image showing a bottle of beer? Choices: ['yes', 'no'] Answer: no",
            "Question: What is in the image? Choices: ['bottle of wine', 'beer', 'bottle of whiskey', 'bottle of vodka'] Answer: bottle of wine",
            "Question: Is there a beer in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a bottle of wine in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: How many bottles of wine are there in the image? Choices: ['one', 'two', 'three', 'four'] Answer: one",
            "Question: Does the image look deformed? Choices: ['slightly', 'moderately', 'severely'] Answer: slightly",
        ],
    },
    {
        "Input": "A computer mouse. Not a mouse.",
        "Output": [
            "Question: What is in the image? Choices: ['mouse', 'computer mouse', 'keyboard', 'monitor'] Answer: computer mouse",
            "Question: Is there a computer mouse in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is there a mouse in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: What type of mouse is in the image? Choices: ['computer mouse', 'animal mouse', 'both', 'none'] Answer: computer mouse",
            "Question: Does the object in the image appear deformed? Choices: ['slightly', 'moderately', 'severely'] Answer: slightly",
        ],
    },
    {
        "Input": "A desk lamp. Not a lamp.",
        "Output": [
            "Question: What is in the image? Choices: ['lamp', 'desk lamp', 'table', 'chair'] Answer: desk lamp",
            "Question: Is there a lamp in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a desk lamp in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: What type of lamp is in the image? Choices: ['desk lamp', 'floor lamp', 'wall lamp', 'none'] Answer: desk lamp",
            "Question: Does the image look deformed? Choices: ['slightly', 'moderately', 'severely'] Answer: slightly",
        ],
    },
    {
        "Input": "A red umbrella. Not blue.",
        "Output": [
            "Question: Is the umbrella in the image blue? Choices: ['yes', 'no'] Answer: no",
            "Question: Is the umbrella in the image red? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is there a blue umbrella in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: What is the color of the umbrella in the image? Choices: ['red', 'blue', 'green', 'yellow'] Answer: red",
            "Question: What type of object is in the image? Choices: ['umbrella', 'parasol', 'canopy', 'awning'] Answer: umbrella",
            "Question: Is the red umbrella in the image clearly visible? Choices: ['completely visible', 'partially visible', 'barely visible', 'not visible'] Answer: completely visible",
            "Question: Does the image look deformed? Choices: ['slightly', 'moderately', 'severely'] Answer: slightly",
        ],
    },
    {
        "Input": "A white wooden bench. Not a stone bench.",
        "Output": [
            "Question: What is the material of the bench in the image? Choices: ['wooden', 'stone', 'metal', 'plastic'] Answer: wooden",
            "Question: Is there a stone bench? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a wooden bench? Choices: ['yes', 'no'] Answer: yes",
            "Question: What color is the bench in the image? Choices: ['white', 'black', 'brown', 'gray'] Answer: white",
            "Question: Is there a bench made of stone? Choices: ['yes', 'no'] Answer: no",
            "Question: Is the object in the image unrecognizable due to deformation? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the object in the image appear deformed? Choices: ['slightly', 'moderately', 'severely'] Answer: slightly",
        ],
    },
    {
        "Input": "A green apple. Not red.",
        "Output": [
            "Question: What is the color of the apple in the image? Choices: ['green', 'red', 'yellow', 'blue', 'none'] Answer: green",
            "Question: Is the apple in the image red? Choices: ['yes', 'no'] Answer: no",
            "Question: Is the apple in the image green? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is there a red apple in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: How many green apples are there in the image? Choices: ['one', 'two', 'three', 'four', 'none'] Answer: one",
            "Question: What type of fruit is in the image? Choices: ['apple', 'banana', 'orange', 'pear', 'none'] Answer: apple",
            "Question: Does the object in the image appear deformed? Choices: ['slightly', 'moderately', 'severely'] Answer: slightly",
        ],
    },
    {
        "Input": "A yellow car. Not blue.",
        "Output": [
            "Question: Is there a yellow car in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: Does the image show a blue car? Choices: ['yes', 'no'] Answer: no",
            "Question: What is the color of the car in the image? Choices: ['yellow', 'blue', 'red', 'green', 'black'] Answer: yellow",
            "Question: Is the car in the image blue? Choices: ['yes', 'no'] Answer: no",
            "Question: Is the car in the image yellow? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is there a blue car in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: How many yellow cars are there in the image? Choices: ['one', 'two', 'three', 'more than three', 'none'] Answer: one",
            "Question: What type of vehicle is in the image? Choices: ['car', 'truck', 'motorcycle', 'bicycle', 'none'] Answer: car",
            "Question: Does the image look deformed? Choices: ['slightly', 'moderately', 'severely'] Answer: slightly",
        ],
    },
]


OBJECT_COUNT_PROMPT = [
    {
        "Input": "A photo of a dog.",
        "Output": [
            "Question: Does the image show exactly one dog? Choices: ['yes', 'no'] Answer: yes",
            "Question: How many animals are there in the image? Choices: ['one', 'more than one', 'none'] Answer: one",
            "Question: How many dogs are visible in the image? Choices: ['one', 'more than one', 'none'] Answer: one",
            "Question: How many dogs are there in the image? Choices: ['one', 'two', 'three', 'more than three', 'none'] Answer: one",
            "Question: Is there a dog present in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: What is the animal in the image? Choices: ['dog', 'cat', 'wolf'] Answer: dog",
            "Question: Is the animal in the image a dog? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is the kite in the image clearly visible? Choices: ['completely visible', 'partially visible', 'barely visible', 'not visible'] Answer: completely visible",
        ],
    },
    {
        "Input": "A photo of a bird.",
        "Output": [
            "Question: Does the image show exactly one bird? Choices: ['yes', 'no'] Answer: yes",
            "Question: How many birds are visible in the image? Choices: ['one', 'more than one', 'none'] Answer: one",
            "Question: How many birds are there in the image? Choices: ['one', 'two', 'three', 'none'] Answer: one",
            "Question: Is the only one bird in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: What do you see in the image? Choices: ['bird', 'fish'] Answer: bird",
            "Question: What is in the image? Choices: ['bird', 'fish'] Answer: bird",
            "Question: Is there a bird in the image? Choices: ['yes', 'no'] Answer: yes",
        ],
    },
    {
        "Input": "A photo of a horse.",
        "Output": [
            "Question: Does the image show exactly one horse? Choices: ['yes', 'no'] Answer: yes",
            "Question: How many horses are there in the image? Choices: ['one', 'two', 'three', 'none'] Answer: one",
            "Question: How many horses are visible in the image? Choices: ['one', 'more than one', 'none'] Answer: one",
            "Question: What is the animal in the image? Choices: ['horse', 'cow'] Answer: horse",
            "Question: What do you see in the image? Choices: ['horse', 'cow'] Answer: horse",
            "Question: Is there a horse in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is the horse in the image clearly visible? Choices: ['yes', 'no'] Answer: yes",
        ],
    },
    {
        "Input": "Two dogs",
        "Output": [
            "Question: How many dogs are there in the image? Choices: ['one', 'two', 'three', 'none'] Answer: two",
            "Question: What type of animal is in the image? Choices: ['dog', 'cat'] Answer: dog",
            "Question: What do you see in the image? Choices: ['one dog', 'two dogs', 'three dogs', 'none'] Answer: two dogs",
            "Question: Are there dogs in the image? Choices: ['yes', 'no'] Answer: yes",
        ],
    },
    {
        "Input": "Two sandwiches",
        "Output": [
            "Question: How many sandwiches are there in the image? Choices: ['one', 'two', 'three', 'none'] Answer: two",
            "Question: What type of food is in the image? Choices: ['sandwich', 'pizza'] Answer: sandwich",
            "Question: What do you see in the image? Choices: ['one sandwich', 'two sandwiches', 'three sandwiches', 'none'] Answer: two sandwiches",
            "Question: Are there sandwiches in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: Does the image look deformed? Choices: ['slightly', 'moderately', 'severely'] Answer: slightly",
        ],
    },
    {
        "Input": "A photo of a kite.",
        "Output": [
            "Question: Does the image show exactly one kite? Choices: ['yes', 'no'] Answer: yes",
            "Question: How many kites are visible in the image? Choices: ['one', 'more than one', 'none'] Answer: one",
            "Question: Is the kite overlapping with other objects in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there only one kind of kite in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: What do you see in the image? Choices: ['kite', 'balloon', 'bird'] Answer: kite",
            "Question: What is in the image? Choices: ['kite', 'balloon', 'plane'] Answer: kite",
            "Question: Is there a balloon in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a kite in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is the kite in the image clearly visible? Choices: ['completely visible', 'partially visible', 'barely visible', 'not visible'] Answer: completely visible",
        ],
    },
    {
        "Input": "An image of a bottle of wine.",
        "Output": [
            "Question: How many bottles of wine are there in the image? Choices: ['one', 'two', 'three', 'none'] Answer: one",
            "Question: Does the image show exactly one bottle of wine? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is the wine bottle clearly visible in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: What type of drink is in the image? Choices: ['wine', 'beer', 'whiskey', 'vodka'] Answer: wine",
            "Question: Is the image showing a bottle of beer? Choices: ['yes', 'no'] Answer: no",
            "Question: What is in the image? Choices: ['bottle of wine', 'beer', 'bottle of whiskey', 'bottle of vodka'] Answer: bottle of wine",
            "Question: Is there a bottle of wine in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: Does the image look deformed? Choices: ['slightly', 'moderately', 'severely'] Answer: slightly",
        ],
    },
    {
        "Input": "An image shows a computer mouse.",
        "Output": [
            "Question: How many computer mice are visible in the image? Choices: ['0', '1', '2', '3', 'more than 3'] Answer: '1'",
            "Question: Is a computer mouse clearly visible in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: Does the computer mouse in the image appear to be damaged or deformed in any way? Choices: ['yes', 'no'] Answer: no",
            "Question: What is the main object in the image? Choices: ['keyboard', 'computer mouse', 'both', 'neither'] Answer: computer mouse",
            "Question: Is there a computer mouse present in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is the computer mouse in the image being used by someone? Choices: ['yes', 'no'] Answer: no",
            "Question: How visible is the computer mouse in the image? Choices: ['completely visible', 'partially visible', 'barely visible', 'not visible'] Answer: completely visible",
        ],
    },
    {
        "Input": "A photo of a desk lamp.",
        "Output": [
            "Question: How many desk lamps are visible in the image? Choices: ['one', 'more than one', 'none'] Answer: one",
            "Question: Does the desk lamp in the image appear to be deformed in any way? Choices: ['yes', 'no'] Answer: no",
            "Question: Is the desk lamp clearly visible in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: What is in the image? Choices: ['desk lamp', 'kerosine lamp'] Answer: lamp",
            "Question: Do you see a desk lamp in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a desk lamp in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is the desk lamp placed on a desk in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is the desk lamp in the image clearly visible? Choices: ['completely visible', 'partially visible', 'barely visible', 'not visible'] Answer: completely visible",
        ],
    },
    {
        "Input": "A photo of a red umbrella.",
        "Output": [
            "Question: How many red umbrellas are visible in the image? Choices: ['one', 'more than one', 'none'] Answer: one",
            "Question: Is there more than one umbrella in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: What is the main object in the image? Choices: ['umbrella', 'raincoat', 'boot'] Answer: umbrella",
            "Question: What is the color of the umbrella in the image? Choices: ['red', 'blue', 'green', 'yellow'] Answer: red",
            "Question: Is the umbrella in the image red? Choices: ['yes', 'no'] Answer: yes",
            "Question: What type of object is in the image? Choices: ['umbrella', 'parasol', 'canopy', 'awning'] Answer: umbrella",
            "Question: Is the red umbrella in the image clearly visible? Choices: ['yes', 'no'] Answer: yes",
            "Question: Does the image look deformed? Choices: ['yes', 'no'] Answer: no",
        ],
    },
    {
        "Input": "An image of a white wooden bench.",
        "Output": [
            "Question: How many benches are there in the image? Choices: ['one', 'two', 'three', 'more than three', 'none'] Answer: one",
            "Question: What is the material of the bench in the image? Choices: ['wooden', 'stone', 'metal', 'plastic'] Answer: wooden",
            "Question: Is there a wooden bench? Choices: ['yes', 'no'] Answer: yes",
            "Question: What color is the bench in the image? Choices: ['white', 'black', 'brown', 'gray'] Answer: white",
            "Question: Is the object in the image unrecognizable due to deformation? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the object in the image appear deformed? Choices: ['yes', 'no'] Answer: no",
        ],
    },
    {
        "Input": "An image shows an green apple.",
        "Output": [
            "Question: How many green apples are there in the image? Choices: ['one', 'two', 'three', 'more than three', 'none'] Answer: one",
            "Question: Does the image show exactly one green apple? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is the green apple clearly visible in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: What is the color of the apple in the image? Choices: ['green', 'red', 'both', 'none'] Answer: green",
            "Question: Is the apple in the image green? Choices: ['yes', 'no'] Answer: yes",
            "Question: What is the color of the fruit in the image? Choices: ['green', 'red', 'yellow'] Answer: green",
            "Question: Is there more than one apple in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the image look deformed? Choices: ['slightly', 'moderately', 'severely'] Answer: slightly",
        ],
    },
    {
        "Input": "A photo of a yellow car.",
        "Output": [
            "Question: How many yellow cars are there in the image? Choices: ['one', 'two', 'three', 'more than three', 'none'] Answer: one",
            "Question: Is there a yellow car in the image? Choices: ['yes', 'no'] Answer: yes",
            "Question: What is the color of the car in the image? Choices: ['yellow', 'blue', 'red', 'green'] Answer: yellow",
            "Question: Is the car in the image yellow? Choices: ['yes', 'no'] Answer: yes",
            "Question: Is there more than one yellow car in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: What type of vehicle is in the image? Choices: ['car', 'truck', 'motorcycle', 'bicycle', 'none'] Answer: car",
            "Question: Does the image look deformed? Choices: ['slightly', 'moderately', 'severely'] Answer: slightly",
        ],
    },
]


OBJECT_ABSENT_PROMPT = [
    {
        "Input": "Not a cat.",
        "Output": [
            "Question: Is the animal in the image a cat? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the image show a cat? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a cat in the image? Choices: ['yes', 'no'] Answer: no",
        ],
    },
    {
        "Input": "Not a fish.",
        "Output": [
            "Question: Is there a fish in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the image show a fish? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a fish in the image? Choices: ['yes', 'no'] Answer: no",
        ],
    },
    {
        "Input": "Not a cow.",
        "Output": [
            "Question: Is there a cow in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the image show a cow? Choices: ['yes', 'no'] Answer: no",
        ],
    },
    {
        "Input": "Not a balloon.",
        "Output": [
            "Question: Is there a balloon in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the image show a balloon? Choices: ['yes', 'no'] Answer: no",
        ],
    },
    {
        "Input": "Not beer.",
        "Output": [
            "Question: Is there a beer in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the image show a beer? Choices: ['yes', 'no'] Answer: no",
        ],
    },
    {
        "Input": "Not a computer mouse.",
        "Output": [
            "Question: Is there a computer mouse in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the image show a computer mouse? Choices: ['yes', 'no'] Answer: no",
        ],
    },
    {
        "Input": "Not a general lamp.",
        "Output": [
            "Question: Is there a lamp in the image? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the image show a lamp? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a general lamp in the image? Choices: ['yes', 'no'] Answer: no",
        ],
    },
    {
        "Input": "Not a blue umbrella.",
        "Output": [
            "Question: Is the umbrella in the image blue? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the image show a blue umbrella? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a blue umbrella in the image? Choices: ['yes', 'no'] Answer: no",
        ],
    },
    {
        "Input": "Not a stone bench.",
        "Output": [
            "Question: Is there a stone bench? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the image show a stone bench? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a bench made of stone? Choices: ['yes', 'no'] Answer: no",
        ],
    },
    {
        "Input": "Not a red apple.",
        "Output": [
            "Question: Is the apple in the image red? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the image show a red apple? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a red apple in the image? Choices: ['yes', 'no'] Answer: no",
        ],
    },
    {
        "Input": "Not a blue car.",
        "Output": [
            "Question: Is the car in the image blue? Choices: ['yes', 'no'] Answer: no",
            "Question: Does the image show a blue car? Choices: ['yes', 'no'] Answer: no",
            "Question: Is there a blue car in the image? Choices: ['yes', 'no'] Answer: no",
        ],
    },
]
