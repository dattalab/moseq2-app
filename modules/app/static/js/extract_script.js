$(document).onchange(function() {
    $("#extract").load("../templates/extract.html");
});

var benchFiles = []
var openNavBool = false;

function getLocalDir() {
    $.get('/get-local-dir', {}, function(resp) {
        if (resp.ok) {
            // change button color to green and display extract button
            var fileList = resp.files;
            benchFiles = fileList;
            for(var i = 0; i < fileList.length; i++) {
                $('#filelist').append('<option>'+fileList[i]+'</option>');
                if(fileList[i] == 'session1/depth.dat') {
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
            //Change to sessions
            //selDiv.innerHTML += "Total number of files identified: " + files.length + " <br/>"
            console.log('config successfully generated');

        })
        .catch(() => {
            console.log('error :(');
            // Error: inform user of upload error response.
    });
}

function extractRaw() {
    $.get('/extract-raw', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
            window.location.reload();
        } else {
            alert(resp.message);
        }
    });
}

function findROI() {
    $.get('/find-roi', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
            window.location.reload();
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