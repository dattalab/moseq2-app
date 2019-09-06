$(document).onchange(function() {
    $("#extract").load("../templates/extract.html");
});

var benchFiles = []
var openNavBool = false;

window.onload = function() {

}


function loadImageModal(event) {

    // Get the modal
    var modal = document.getElementById("imageModal");
    // Get the <span> element that closes the modal
    var span = document.getElementsByClassName("close")[0];

    // Get the image and insert it inside the modal - use its "alt" text as a caption
    var img = document.getElementById(event.target.id);

    var modalImg = document.getElementById("img01");
    var captionText = document.getElementById("caption");
    img.onclick = function(){
      modal.style.display = "block";
      modalImg.src = img.src;
      captionText.innerHTML = this.alt;
    }

    // When the user clicks on <span> (x), close the modal
    span.onclick = function() {
      modal.style.display = "none";
    }

    // When the user clicks anywhere outside of the modal, close it
    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }
}

function getLocalDir() {
    $.get('/get-local-dir', {}, function(resp) {
        if (resp.ok) {
            // change button color to green and display extract button
            var fileList = resp.files;
            benchFiles = fileList;
            var e = document.querySelector("#filelist");
            var child = e.lastElementChild;
            while (child) {
                e.removeChild(child);
                child = e.lastElementChild;
            }
            var f = document.querySelector("#extractedlist");
            var child = f.lastElementChild;
            while (child) {
                f.removeChild(child);
                child = f.lastElementChild;
            }
            for(var i = 0; i < fileList.length; i++) {
                $('#filelist').append('<option onclick="selectFile(event)">'+fileList[i]+'</option>');
                if(fileList[i] == 'sample1/depth.dat') {
                    $('#extractedlist').append('<option>'+fileList[i]+'</option>');
                }
            }
        } else {
            alert(resp.message);
        }
    });
}

function uploadExParams() {
    var formData = new FormData();
    let url = '/generate-config'
    $('#extract-params input, #extract-params select').each(
        function(index){
            var input = $(this);
            if (formData.get(input.attr('name'))) {
                var prev = formData.get(input.attr('name'));
                let newVal = [prev, input.val()];
                formData.set(input.attr('name'), newVal);
            } else{
                formData.append(input.attr('name'), input.val());
            }
        }
    );

    fetch(url, {
            method: 'POST',
            body: formData
        })
        .then((resp) => {
            // Done: inform user that upload is finished.
            console.log('config successfully generated');

        })
        .catch(() => {
            console.log('error :(');
            // Error: inform user of upload error response.
    });
}

function extractRaw() {
    var url = '/extract-raw'
    var formData = new FormData();

    var ul = document.getElementById("");

    $('#selected-list li').each(function(i)
    {
       var filename = $(this).attr('id');
       filename = filename.replace('-', '/');

       formData.append('depth-file', filename);
    });


    fetch(url, {
            method: 'POST',
            body: formData
        })
        .then((resp) => {
            // Done: inform user that upload is finished.
            //Change to sessions
            //selDiv.innerHTML += "Total number of files identified: " + files.length + " <br/>"
            var photos = ['/static/output_imgs/bground.tiff', '/static/output_imgs/roi_00.tiff', '/static/output_imgs/first_frame.tiff', '/static/output_imgs/results_00.mp4'];
            var alts = ['Background Image', 'ROI Image', 'First Frame', 'Extracted Video'];
            var gallery = document.getElementById('gal');
            for(var i = 0; i < photos.length; i++) {
                var figure = document.createElement("figure");
                figure.setAttribute('class','gallery__item');

                if(i < photos.length-1) {
                    var img = document.createElement("img");
                    img.setAttribute('class', "gallery__img");
                    img.setAttribute('src', photos[i]);
                    img.setAttribute('id', "img"+i);
                    img.setAttribute('type', 'image/tiff');
                    img.setAttribute('alt', alts[i]);
                    figure.appendChild(img)
                }
                else {
                    var vid = document.createElement("video");
                    vid.setAttribute('class', "gallery__img");
                    vid.setAttribute('alt', alts[i]);
                    vid.setAttribute('id', "vid"+i);
                    var source = document.createElement("source");
                    source.setAttribute('src', photos[i]);
                    source.setAttribute('type', 'video/mp4');
                    vid.appendChild(source);
                    figure.appendChild(vid)
                }
                gallery.appendChild(figure);
            }
            console.log('extraction successfully performed');

        })
        .catch(() => {
            console.log('error :(');
            // Error: inform user of upload error response.
    });

}

function findROI() {
    $.get('/find-roi', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
        } else {
            alert(resp.message);
        }
    });
}

function copySlices() {
    $.get('/copy-slice', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);

        } else {
            alert(resp.message);
        }
    });
}


function displayFlipModal(event) {
    // Get the modal
    var modal = document.getElementById("flipModal");

    // Get the <span> element that closes the modal
    var span = document.getElementsByClassName("close")[2];

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
      }
    }
}

function downloadFlip(event) {
    var clickedId = event.target.id;
    $.get('/download-flip-file', {"flip-id": clickedId}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
        } else {
            alert(resp.message);
        }
    });
}

function toggleBatchModeExtract(evt) {
    if (evt.currentTarget.className == "choice") {
        evt.currentTarget.className += " active";
        console.log(evt.currentTarget);
        var d = document.getElementsByClassName('choice');
        for(var i = 0; i < d.length; i++) {
            if (d[i].id == 'extract-button') {
                console.log(d[i].id);
                document.getElementById(d[i].id).innerHTML = 'Batch Extract';
                document.getElementById(d[i].id).value = 'batch-extract';
                document.getElementById(d[i].id).style.borderRight = '3px solid orange';
            }
        }
    } else {
        evt.currentTarget.className = evt.currentTarget.className.replace(" active", "");

        var d = document.getElementsByClassName('choice');
        for(var i = 0; i < d.length; i++) {
            if (d[i].id == 'extract-button') {
                console.log(d[i].id);
                document.getElementById(d[i].id).value = 'extract-button';
                document.getElementById(d[i].id).innerHTML = 'Extract Raw Data';
                document.getElementById(d[i].id).style.borderRight = '';
            }
        }
    }
}

function openNav() {
  if (!openNavBool) {
      getLocalDir();
      document.getElementById("mySidebar").style.width = "300px";
      document.getElementById("extract-tab").style.marginRight = "300px";
      openNavBool = !openNavBool;
  } else {
      document.getElementById("mySidebar").style.width = "0";
      document.getElementById("extract-tab").style.marginRight = "0";
      openNavBool = !openNavBool;
  }
}

function closeNav() {
  document.getElementById("mySidebar").style.width = "0";
  document.getElementById("extract-tab").style.marginRight = "0";
  openNavBool = !openNavBool;
}

function openTab(evt, tabName) {

    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("choice-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }

    tablinks = document.getElementsByClassName("choice");
    for(i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
    if (evt.currentTarget.id == 'init-step-button') {
        getLocalDir();
    }
}

function selectFile(event) {
    var clickedOpt = event.target;
    var clickedOptName = event.target.innerHTML;
    var ul = document.getElementById("selected-list");
    var clickedOptId = clickedOptName.replace('/','-');
    //clickedOptId = clickedOptId.replace('.', '_');

    if (clickedOpt.selected) {
        var li = document.createElement("li");
        li.setAttribute('id',clickedOptId);
        li.appendChild(document.createTextNode(clickedOptName));
        ul.appendChild(li);
    } else {
        var item = document.getElementById(clickedOptId);
        ul.removeChild(item);
    }
}