
//
// javascript for sketchpad & information displays in ANN.html
//

// toggle between show and hide while keeping space
function toggleVisibility(scenario) {
    var x = document.getElementById(scenario);
    if (x.style.visibility === "hidden") {
        x.style.visibility = "visible";
    } else {
        x.style.visibility = "hidden";
    }
};

//save image to local storage so can be retrieved after a screen refresh
function store_sketch() {
    localStorage.setItem("sketchpad", canvas.toDataURL());
}

// Variables for referencing the canvas and 2dcanvas context
var canvas, ctx;

// Variables to keep track of the mouse position and left-button status 
var mouseX, mouseY, mouseDown = 0;

// Variables to keep track of the touch position
var touchX, touchY;

// Keep track of the old/last position when drawing a line
// We set it to -1 at the start to indicate that we don't have a good value 
// for it yet
var lastX, lastY = -1;

// Draws a line between the specified position on the supplied canvas name
// Parameters are: A canvas context, the x position, the y position, 
// the size of the dot

function drawLine(ctx, x, y, size) {

    // If lastX is not set, set lastX and lastY to the current position 
    if (lastX == -1) {
        lastX = x;
        lastY = y;
    }

    // Let's use black by setting RGB values to 0, and 255 alpha (completely opaque)
    r = 0; g = 0; b = 0; a = 255;

    // Select a fill style
    //ctx.strokeStyle = "rgba(" + r + "," + g + "," + b + "," + (a / 255) + ")";
    ctx.strokeStyle = 'black'
    // Set the line "cap" style to round, so lines at different angles 
    // can join into each other
    ctx.lineCap = "round";
    ctx.lineJoin = "round";


    // Draw a filled line
    ctx.beginPath();

    // First, move to the old (previous) position
    ctx.moveTo(lastX, lastY);

    // Now draw a line to the current touch/pointer position
    ctx.lineTo(x, y);

    // Set the line thickness and draw the line
    ctx.lineWidth = size;
    ctx.stroke();

    ctx.closePath();

    // Update the last position to reference the current position
    lastX = x;
    lastY = y;
}

// Clear the canvas context using the canvas width and height
function clearCanvas(canvas, ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById("myprediction").innerHTML =
        "<h2 style='font-size:12vw'>&nbsp;&nbsp;-</h2>";
    document.getElementById("prediction2").innerHTML =
        "<h5>&nbsp;&nbsp;</h5>";
    toggleVisibility("tbodyid");
}

// Keep track of the mouse button being pressed and draw dot
function sketchpad_mouseDown() {
    mouseDown = 1;
    drawLine(ctx, mouseX, mouseY, 10);
}

// Keep track of the mouse button being released
function sketchpad_mouseUp() {
    mouseDown = 0;

    // Reset lastX and lastY to -1 to indicate that they are now invalid
    lastX = -1;
    lastY = -1;
}

// Keep track of the mouse position and draw a dot if button is pressed
function sketchpad_mouseMove(e) {
    // Update the mouse co-ordinates when moved
    getMousePos(e);

    // Draw a dot if the mouse button is currently being pressed
    if (mouseDown == 1) {
        drawLine(ctx, mouseX, mouseY, 10);
    }
}

// Get the current mouse position relative to the top-left of the canvas
function getMousePos(e) {
    if (!e)
        var e = event;

    if (e.offsetX) {
        mouseX = e.offsetX;
        mouseY = e.offsetY;
    }
    else if (e.layerX) {
        mouseX = e.layerX;
        mouseY = e.layerY;
    }
}

// Draw something when a touch start is detected
function sketchpad_touchStart() {
    // Update the touch co-ordinates
    getTouchPos();

    drawLine(ctx, touchX, touchY, 10);

    // Prevents an additional mousedown event being triggered
    event.preventDefault();
}

function sketchpad_touchEnd() {
    // Reset lastX and lastY to -1 to indicate that they are now invalid, 
    lastX = -1;
    lastY = -1;
}

// Draw something and prevent the default scrolling when touch movement is detected
function sketchpad_touchMove(e) {
    // Update the touch co-ordinates
    getTouchPos(e);

    drawLine(ctx, touchX, touchY, 10);    // 10 is thickness of line

    // Prevent a scrolling action as a result of this touchmove triggering.
    event.preventDefault();
}

// Get the touch position relative to the top-left of the canvas
function getTouchPos(e) {
    if (!e)
        var e = event;

    if (e.touches) {
        if (e.touches.length == 1) { // Only deal with one finger
            var touch = e.touches[0]; // Get the information for finger #1
            touchX = touch.pageX - touch.target.offsetLeft;
            touchY = touch.pageY - touch.target.offsetTop;
        }
    }
}


// Set-up the canvas and add event handlers after the page has loaded
function init() {

    $("html, body").animate({ scrollTop: 0 }, "slow");

    // Get the specific canvas element from the HTML document
    canvas = document.getElementById('sketchpad');

    // If browser supports canvas tag, get the 2d drawing context 
    if (canvas.getContext)
        ctx = canvas.getContext('2d');

    // check for image in local storage; if yes, show on sketchpad    
    if (localStorage.getItem("sketchpad") === null) {
        console.log("local storage empty");
    } else {
        var dataURL = localStorage.getItem("sketchpad");
        var img = new Image;
        img.src = dataURL;
        img.onload = function () {
            ctx.drawImage(img, 0, 0);
        }
    }

    // Check for valid context to draw on before adding event handlers
    if (ctx) {
        // React to mouse events on the canvas
        canvas.addEventListener('mousedown', sketchpad_mouseDown, false);
        canvas.addEventListener('mousemove', sketchpad_mouseMove, false);
        window.addEventListener('mouseup', sketchpad_mouseUp, false);

        // React to touch events on the canvas
        canvas.addEventListener('touchstart', sketchpad_touchStart, false);
        canvas.addEventListener('touchend', sketchpad_touchEnd, false);
        canvas.addEventListener('touchmove', sketchpad_touchMove, false);
    }

}

// Get image
function getImage(canvas, ctx) {

    // get image from sketchpad
    var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    console.log(imageData.data);

    var px = imageData.data;

    var result = null;
    var counter = 0;
    var last = px.length - 1;
    for (var i = 0; i < px.length; i++) {
        if (i === last && counter !== 0 && counter === last) {
            result = false;
            break;
        } else if (px[i] !== 0 && px[i] > 0) {
            result = true;
            break;
        } else {
            counter++;
            continue;
        }
    }

    if (result) {
        console.log('Drawn On');
    } else {
        console.log('Blank');
        document.getElementById("errorfield").innerHTML = "Please sketch a number!";
    };

    //preprocess image (only use darkness reading)
    var px_new = [];
    var len = px.length;

    console.log('len =');
    console.log(len);

    for (i = 0; i < len; i += 4) {
        px_new.push(px[i + 3]);
    }
    var len_new = px_new.length
    console.log(len_new);
    var image = px_new;

    // assign sketchpad image to 'fake_input' so it can be accessed by Flask/Python    
    let element = document.getElementById('fake_input');
    element.value = image;

    store_sketch();

    return result;
}
