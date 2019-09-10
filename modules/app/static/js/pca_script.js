$(document).onload(function(){
    $("#pca").load("../templates/pca.html");
});

function openTabPCA(evt, tabName) {
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

    if (tabName == "gallery-tab") {
        loadGrid();
    }
}

function loadGrid() {
    getLocalDir();
    var grid = document.getElementsByClassName('pca-grid-container')[0];

    var lists = $('#mySidebar select');
    var files;
    for(var i = 0; i < lists.length; i++) {
        if (lists[i].id == "pcalist") {
            files = $('#'+lists[i].id).children();
        }
    }

    var imgFiles = [];
    for(var j=0; j < files.length; j++) {
        if (files[j].innerHTML.includes('.png')){
            imgFiles.push("/static/img/pca/"+files[j].innerHTML.replace('_pca/', ""))
        }
    }

    for(var i = 0; i < imgFiles.length; i++) {
        var filename = imgFiles[i].split('/');
        filename = filename[filename.length-1];

        var computedDiv = document.createElement('div');
        computedDiv.setAttribute('style', 'justify-content:center;')
        computedDiv.setAttribute('class', 'pca-images');

        var title = document.createElement('a');
        title.setAttribute('class', 'img-title');
        var t = document.createTextNode(filename);
        title.appendChild(t);
        computedDiv.appendChild(title);

        var aImg = document.createElement('a');
        aImg.setAttribute('class', 'pca-img');
        computedDiv.appendChild(aImg);

        var imgHolder = document.createElement('div');
        imgHolder.setAttribute('class', 'img-holder');
        aImg.appendChild(imgHolder);

        var img = document.createElement('img');
        img.setAttribute('src', imgFiles[i]);
        img.setAttribute('id', imgFiles[i].replace('/','_'));
        imgHolder.appendChild(img);

        var downloadingImage = new Image();
        downloadingImage.onload = function(){
            img.src = this.src;
        };

        downloadingImage.src = imgFiles[i];
        console.log(imgFiles[i]);
        grid.appendChild(computedDiv);
    }

}

function trainPCA() {

    var url = '/train-pca';
    var formData = new FormData();

    $('#train-list li').each(function(i)
    {
       var filename = $(this).attr('id');
       filename = filename.replace('-', '/');

       formData.append('session-dir', filename);
    });

    $('#train-params input, #train-params select').each(
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
        .then(response => response.json())
        .then(response => {

        })
        .catch(() => {
            console.log('error :(');
            // Error: inform user of upload error response.
    });
}

function clipScores() {
    $.get('/clip-pca-scores', {}, function(resp) {
        if (resp.ok) {
            alert(resp.message);
        } else {
            alert(resp.message);
        }
    });
}


function applyPCA() {
    var url = '/apply-pca';
    var formData = new FormData();

    $('#apply-list li').each(function(i)
    {
       var filename = $(this).attr('id');
       filename = filename.replace('-', '/');

       formData.append('pca-file', filename);
    });

    $('#apply-params input, #apply-params select').each(
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
        .then(response => response.json())
        .then(response => {

        })
        .catch(() => {
            console.log('error :(');
            // Error: inform user of upload error response.
    });
}

function computeCP() {
    var url = '/compute-changepoints';
    var formData = new FormData();

    $('#cp-list li').each(function(i)
    {
       var filename = $(this).attr('id');
       filename = filename.replace('-', '/');

       formData.append('pca-scores', filename);
    });

    $('#cp-params input, #cp-params select').each(
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
        .then(response => response.json())
        .then(response => {

        })
        .catch(() => {
            console.log('error :(');
            // Error: inform user of upload error response.
    });
}