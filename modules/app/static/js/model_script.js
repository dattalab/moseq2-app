$(document).onchange(function(){
    $("#model").load("../templates/model.html");
});

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
        //implement auto directory check
    }
}

function toggleBatchModeModel(evt) {
    if (evt.currentTarget.className == "choice") {
        evt.currentTarget.className += " active";

        var d = document.getElementsByClassName('choice');
        for(var i = 0; i < d.length; i++) {
            if (d[i].id == 'learn-model') {
                document.getElementById(d[i].id).value = 'batch-learn-model';
                document.getElementById(d[i].id).innerHTML = 'Learn-Model + Parameter Scan';
                document.getElementById(d[i].id).style.borderRight = '3px solid orange';
            }
        }
    } else {
        evt.currentTarget.className = evt.currentTarget.className.replace(" active", "");

        var d = document.getElementsByClassName('choice');
        for(var i = 0; i < d.length; i++) {
            if (d[i].id == 'learn-model') {
                document.getElementById(d[i].id).value = 'learn-model';
                document.getElementById(d[i].id).innerHTML = 'Train ARHMM';
                document.getElementById(d[i].id).style.borderRight = '';
            }
        }
    }
}

function learnModel() {
    var url = '/learn-model';
    var formData = new FormData();


    $('#learn-list li').each(function(i)
    {
       var filename = $(this).attr('id');
       filename = filename.replace('-', '/');

       formData.append('score-file', filename);
    });


    $('#model-params input, #model-params select').each(
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

function countFrames() {
    var url = '/count-frames';
    var formData = new FormData();


    $('#count-list li').each(function(i)
    {
       var filename = $(this).attr('id');
       filename = filename.replace('-', '/');

       formData.append('video-file', filename);
    });


    $('#count-params input, #count-params select').each(
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
