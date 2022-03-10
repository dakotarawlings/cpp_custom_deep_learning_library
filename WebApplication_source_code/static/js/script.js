/* 
@author: dakota

this script file defines behaviors for the main (index.html) html file
The HTML file contains a canvas element that allows the user to handraw numbers
This file gets an image file from the HTML upon user submit
sends the file to our flask API which predicts the digit via a FFNN model
and this js file then updates the HTML to display the predicted digit
*/

//Define the URL for the flask API endpoint for the handwritten digit image classification model
const API_URL="/predict";

//function that is called when the user hits the submit button
//gets a blob image data file from the canvas, creates an object url, and passes the data into our post request function to make a post request to our API
function onSubmit() {
    //select our canvas element which contains our handwritten digit
    var canvasElement = document.getElementById("canvas");
    //creates a blob object with our image data  and passes data to API via response function that calls our post reqest function
    canvasBlob=canvasElement.toBlob(function(blob){ 
      //create an object url with our blob object
        URLCanvas = URL.createObjectURL(blob,{type: 'text/plain'});
        //console.log(URLCanvas)
        //create a formdata object with our image blob file that is then passed to our post request function
        const formData=new FormData()  
        formData.append('media', blob)    
        PostRequest(formData)      
        //THIS IS IMPORTANT. formatting our image as a jpeg makes our lives much easier because jpegs are in RGB format not RGBA (RGBAs are difficult to conver to black and white)
        }, 'image/jpeg')  
}

//post request function to call our API for digit prediction
function PostRequest(formData){
    //use the fetch method to pass our image data to the API
    fetch(API_URL, {
      method:"POST",
      body: formData,
      headers: {
        "Accept-Encoding": "*",
        "Connection": "keep-alive"
      }
      //define functions that handle API response
        }).then(response => response.json()
        .then(function(data) {
            console.log(data)
            //get the class prediction from the API response
            predictedChar=data['predictions'][0]['label'];
            //call our function that updates the HTML element that displays our predicted class
            updatePrediction(predictedChar)
             }))
}

//Simple function that takes in the class prediction from the API and updates the HTML element that displays the class
function updatePrediction(prediction) {
    var predictionElement=d3.select('#prediction');
    var predictionTemplate=`<span style= "font-weight: bold; font-size: 150px; color: white;"> ${prediction}</span>`;
    predictionElement.html(predictionTemplate)
}

//The following script defines the behavior of our canvas element with a draing feature

// wait for the content of the window elementto load
window.addEventListener('load', ()=>{
    //select our canvas element
    const canvasElement = document.getElementById('canvas')
    //call our start painting function when a mouse down action occurs
    canvasElement.addEventListener('mousedown', startPainting);
    //call our stop painting function when a mouse up action occurs
    canvasElement.addEventListener('mouseup', stopPainting);
    //call our sketch function when a mouse move action occurs
    canvasElement.addEventListener('mousemove', sketch);

    //select our erase button element
    const clearButton = document.getElementById('erase')
    //call our clear canvas function when the erase button is clicked (right now it just reloads the page)
    clearButton.addEventListener('click', clearCanvas);
    
});

//select our canvas element 
const canvas = document.querySelector('#canvas');
   
// Context for the canvas for 2 dimensional operations
const ctx = canvas.getContext('2d');
//set our fill style to white for easy image processing and fill our canvas rectangle
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

//clear canvas function to reload the page when the erase button is clicked
function clearCanvas(event){
    location.reload();
}    
   
// Stores the initial position of the cursor
let coord = {x:0 , y:0}; 
   
// Flag that we are going to use to trigger drawing
let paint = false;
    
// Function to update the coordianates of the cursor when an event e is triggered to the coordinates 
function getPosition(event){
  coord.x = event.clientX - canvas.offsetLeft;
  coord.y = event.clientY - canvas.offsetTop;
}
  
// Functions to toggle the flag to startand stop drawing
function startPainting(event){
  paint = true;
  getPosition(event);
}
function stopPainting(){
  paint = false;
}

//function to dar line between cursor coordinates
function sketch(event){
  //check that our flag is on
  if (!paint) return;
  //direct our context to costruct a line between coordinates
  ctx.beginPath();
  ctx.lineWidth = 20;
   
  // Set the line to black and the end of the lines drawn to a round shape.
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';
      
  //move the cursor to our new position
  ctx.moveTo(coord.x, coord.y);
  //updat the position of the cursor as the mouse moves
  getPosition(event);
   
  //trace a line between mouse coordinates
  ctx.lineTo(coord.x , coord.y);
    
  // Draw the line.
  ctx.stroke();
}

//Event listener for when the user clicks submit
d3.select("#button").on("click",onSubmit);





