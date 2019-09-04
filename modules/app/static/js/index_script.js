$(document).ready(function () {
    $(document).on("scroll", onScroll);
    $("#extract").load("../templates/extract.html");
    $("#pca").load("../templates/pca.html");
    $("#model").load("../templates/model.html");
    $("#viz").load("../templates/viz.html");

    //smoothscroll
    $('a[href^="#"]').on('click', function (e) {
        e.preventDefault();
        $(document).off("scroll");

        $('a').each(function () {
            $(this).removeClass('active');
        });

        $(this).addClass('active');
        var target = this.hash,
            menu = target;
        $target = $(target);
        $('html, body').stop().animate({
            'scrollTop': $target.offset().top-55
        }, 100, 'swing', function () {
            window.location.hash = target;
            $(document).on("scroll", onScroll);
        });
    });
});

window.onload = function(){

    let dropArea = document.getElementById('drop-area');
    let dropArea2 = document.getElementById('drop-area2');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, preventDefaults, false)
      dropArea2.addEventListener(eventName, preventDefaults, false)
    });

    function preventDefaults (e) {
      e.preventDefault();
      e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
      dropArea.addEventListener(eventName, highlight, false)
      dropArea2.addEventListener(eventName, highlight, false)
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, unhighlight, false)
      dropArea2.addEventListener(eventName, unhighlight, false)
    });

    function highlight(e) {
      dropArea.classList.add('highlight');
      dropArea2.classList.add('highlight');
    }

    function unhighlight(e) {
      dropArea.classList.remove('highlight');
      dropArea2.classList.remove('highlight');
    }

    dropArea.addEventListener('drop', handleDrop, true)
    dropArea2.addEventListener('drop', handleDrop, true)

    function handleDrop(e) {
      let dt = e.dataTransfer
      let files = dt.files

      selDiv = document.getElementById("selectedFiles");
      selDiv.innerHTML = "";

      for(var i=0; i<files.length; i++) {
        var f = files[i];
        selDiv.innerHTML += f.name + "<br/>";
      }


      handleFiles(files)
    }

    function handleFiles(files) {
      ([...files]).forEach(uploadFile); //converting the FileList to an array to be handled
    }

    function uploadFile(file) {
        let url = '/uploadFile';
        let formData = new FormData();


        // Check if file is directory
        //console.log(window.FileSystemDirectoryEntry(file));
        formData.append("files", file);
        fetch(url, {
            method: 'POST',
            body: formData
        })
        .then((resp) => {
            // Done: inform user that upload is finished.
            //Change to sessions
            //selDiv.innerHTML += "Total number of files identified: " + files.length + " <br/>"
            console.log(file.name+' uploaded');

        })
        .catch(() => {
            console.log('error :(');
            // Error: inform user of upload error response.
        });
    }
}

function handleFiles(files) {
      ([...files]).forEach(uploadFile); //converting the FileList to an array to be handled
}

function uploadFile(file) {
    let url = '/uploadFile';
    let formData = new FormData();

    formData.append("file", file);
    fetch(url, {
        method: 'POST',
        body: formData
    })
    .then((resp) => {
        // Done: inform user that upload is finished.
        //Change to sessions
        //selDiv.innerHTML += "Total number of files identified: " + files.length + " <br/>"
        console.log(file.name+' uploaded');

    })
    .catch(() => {
        // Error: inform user of upload error reponse.
    });
}

function loadDiv(id) {
    var x = document.getElementById(id);

    var otherDivs = document.getElementsByClassName("acc-div");
    var otherDiv;
    for(var i = 0; i < otherDivs.length; i++) {
        if (otherDivs[i].id != id) {
            otherDiv = document.getElementById(otherDivs[i].id);
            break;
        }
    }
    if (x.className.indexOf("hidden") >= 0) {
        x.className = x.className.replace(" hidden", "");
        if (otherDiv.className.indexOf("hidden") == -1) {
            otherDiv.className += " hidden";
        }
    } else {
        x.className += " hidden";
    }
}


function displayParamModal(ev) {
    // Get the modal
    var modal = document.getElementById("paramModal");

    // Get the <span> element that closes the modal
    var span = document.getElementsByClassName("close")[1];

    // Get command user is requesting parameter modal for
    var helpMethod = document.getElementsByClassName("choice active")[0].innerHTML;
    console.log(document.getElementsByClassName("choice active")[0].baseURI);
    var header = modal.children[0].children[0].children[1].innerHTML = helpMethod;

    // When the user clicks the button, open the modal
    modal.style.display = "block";

    // When the user clicks on <span> (x), close the modal
    span.onclick = function() {
      modal.style.display = "none";
    }

    // When the user clicks anywhere outside of the modal, close it
    window.onclick = function(event) {
      if (event.target == modal) {
        modal.style.display = "none";
        window.location.reload();
      }
    }
}

function displayHelpModal(ev) {
    // Get the modal
    var modal = document.getElementById("helperModal");

    // Get the <span> element that closes the modal
    var span = document.getElementsByClassName("close")[0];

    // Get command user is requesting help information modal for
    var helpMethod = document.getElementsByClassName("choice active")[0].innerHTML;
    var header = modal.children[0].children[0].children[1].innerHTML = helpMethod;

    // When the user clicks the button, open the modal
    modal.style.display = "block";

    // When the user clicks on <span> (x), close the modal
    span.onclick = function() {
      modal.style.display = "none";
    }

    // When the user clicks anywhere outside of the modal, close it
    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
            window.location.reload();
        }
    }
}

function onScroll(event){
    var scrollPos = $(document).scrollTop();
    $('#horizontal-list a').each(function () {
        var currLink = $(this);
        var refElement = $(currLink.attr("href"));
        if (refElement.position().top-70 <= scrollPos && refElement.position().top-70 + refElement.height() > scrollPos) {
            $('#horizontal-list ul li a').removeClass("active");
            currLink.addClass("active");
        }
        else{
            currLink.removeClass("active");
        }
    });
}