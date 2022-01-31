$(document).ready(() => {
    count = parseInt($("#loadMoreBtn").attr('count'));
    remaining = count;
    if (remaining <= 0)
        $('#loadMoreBtn').prop('disabled', true);

    $("#addNewImgBtn").click(() => {
        var csrftoken = document.cookie.match("csrftoken").input.replace('csrftoken=', '');
        var imagesCount = $('div[id^="photoDiv"]').length + 1;
        
        $("#prependDiv").append(`
            <div id="photoDiv` + imagesCount + `" class="col-6 mt-2">
                <div class="card shadow-lg rounded">
                    <div class="card-body text-center">
                        <form id="form` + imagesCount + `" method="post" action="" enctype="multipart/form-data" id="myform">
                            <input type="hidden" name="csrfmiddlewaretoken" value="` + csrftoken + `">
                            <div class="mt-4 col-sm-12">
                                <!-- <input type="file" id="file" class="form-control m-2" name="file" /> -->
                                <div class="uploader">
                                    <img class="img-fluid rounded" id="newImg` + imagesCount + `" src="">
                                    <input type="file" name="file" class="filePhotoClass" id="filePhoto` + imagesCount + `"/>
                                </div>
                            </div>
                        </form>
                    </div>
                </div>
                <div class="card shadow-lg rounded">
                    <div class="card-body text-center">
                        <input type="text" class="form-control mt-2" placeholder="X" name="X" id="X` + imagesCount + `">
                        <input type="text" class="form-control mt-2" placeholder="Y" name="Y" id="Y` + imagesCount + `">
                        <input type="text" class="form-control mt-2" placeholder="Height" name="Latitude" id="Latitude` + imagesCount + `">
                        <input type="text" class="form-control mt-2" placeholder="Width" name="Longitude" id="Longitude` + imagesCount + `">
                    </div>
                </div>
                <div class="card shadow-lg rounded">
                    <div class="card-body text-center">
                        <input type="button" class="btn btn-primary" value="Calculate" id="uploadBtn` + imagesCount + `">
                        <input type="button" class="btn btn-danger" value="Remove" id="removeBtn` + imagesCount + `">
                    </div>
                </div>
            </div>`
        );
    });

    

    $('#loadingDiv')
        .hide()  // Hide it initially
        .ajaxStart(function() {
            $(this).show();
        })
        .ajaxStop(function() {
            $(this).hide();
        })
    ;

    $(document).on("change", "input[id^='filePhoto']", (event) => {
        var id = event.target.id.replace('filePhoto', '');
        var reader = new FileReader();
        reader.onload = (event) => {
            $('#newImg' + id).attr('src', event.target.result);
        }
        reader.readAsDataURL(event.target.files[0]);
    });

    $(document).on("click", "input[id^='uploadBtn']", (event) => {
        var id = event.target.id.replace('uploadBtn', '');

        var fd = new FormData();
        var fdata = new FormData($('#form' + id).get(0));
        var files = $('#filePhoto' + id)[0].files;
        
        if ((files.length > 0 ) && ($("#X" + id).val() && $("#Y" + id).val() && $("#Latitude" + id).val() && $("#Longitude" + id).val())) {
            fd.append('file', files[0]);
            fdata.append('X', $("#X" + id).val());
            fdata.append('Y', $("#Y" + id).val());
            fdata.append('Latitude', $("#Latitude" + id).val());
            fdata.append('Longitude', $("#Longitude" + id).val());
            
            fdata.append('filename', files[0].name)
            $body = $("body");
            $.ajax({
                url: 'predict_lai',
                type: 'POST',
                beforeSend: function() { $body.addClass("loading"); },
                complete: function() { $body.removeClass("loading"); },
                data: fdata,
                contentType: false,
                processData: false,
                success: (response) => {
                    if (response.error != 'err') {
                        if ($("#detectedPhotoDiv" + id).length) {
                            $("#detectedPhotoDiv" + id).remove();
                        }
                        $("#photoDiv" + id).after(`
                            <div class="col-6 mt-4" id="detectedPhotoDiv` + id + `">
                                <div class="card shadow-lg rounded">
                                    <div class="card-body">
                                        <div class="row">
                                            <div class="col-12">
                                                <img class="img-fluid" width="20%" height="20%" alt="detected image" src="` + response.detectedOutputImage + `" id="img1_"` + id + `>
                                                <img class="img-fluid" width="20%" height="20%" alt="masked image" src="` + response.outputImage + `" id="img2_"` + id + `>
                                            </div>
                                            <div class="col-12">
                                                <label class="text-primary" id="LAI"><b class="text-success">LAI is</b> ` + response.LAI + `</label>
                                                <br>
                                                <label class="text-primary" id="FVC"><b class="text-success">FVC is</b> ` + response.FVC + `%</label>
                                                <br>
                                                <label class="text-dark" id="LAI"><b>Image metadata</b></label>
                                                <br>
                                                <label class="text-primary" id="LAI"><b class="text-secondary">X:</b> ` + response.meta[0] + `</label>
                                                <br>
                                                <label class="text-primary" id="LAI"><b class="text-secondary">Y:</b> ` + response.meta[1] + `</label>
                                                <br>
                                                <label class="text-primary" id="LAI"><b class="text-secondary">Height:</b> ` + response.meta[2] + `</label>
                                                <br>
                                                <label class="text-primary" id="LAI"><b class="text-secondary">Width:</b> ` + response.meta[3] + `</label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `);
                    }
                    else {
                        alert(response.message);
                    }
                },
            });
        }
        else {
            alert("Please, choose a file");
        }
    });

    $(document).on("click", "input[id^='removeBtn']", (event) => {
        var id = event.target.id.replace('removeBtn', '');
        
        $("#photoDiv" + id).remove();
        if ($("#detectedPhotoDiv" + id).length)
            $("#detectedPhotoDiv" + id).remove();
    });

    $("#loadMoreBtn").click(() => {
        var offset = $("#loadMoreBtn").attr('offset');
        var csrftoken = document.cookie.match("csrftoken").input.replace('csrftoken=', '');
        $body = $("body");
        $.ajax({
            url: 'load_more_images/?offset=' + offset,
            type: 'GET',
            beforeSend: function() { $body.addClass("loading"); },
            complete: function() { $body.removeClass("loading"); },
            contentType: false,
            processData: false,
            success: (response) => {
                if (response.error != 'err') {
                    $("#loadMoreBtn").attr('offset', parseInt(offset) + 5);
                    count = parseInt($("#loadMoreBtn").attr('count'));
                    remaining = count - 5;

                    response.images.forEach((image, index) => {
                        imageIndex = index + 1;
                        $("#prependDiv").append(`
                            <div id="photoDiv` + imageIndex + `" class="col-6 mt-2">
                                <div class="card shadow-lg rounded">
                                    <div class="card-body text-center">
                                        <form id="form` + imageIndex + `" method="post" action="" enctype="multipart/form-data" id="myform">
                                            <input type="hidden" name="csrfmiddlewaretoken" value="` + csrftoken + `">
                                            <div class="mt-4 col-sm-12">
                                                <!-- <input type="file" id="file" class="form-control m-2" name="file" /> -->
                                                <div class="uploader">
                                                    <img class="img-fluid rounded" id="newImg` + imageIndex + `" src="` + image.image_file + `">
                                                    <input type="file" name="file" class="filePhotoClass" id="filePhoto` + imageIndex + `"/>
                                                </div>
                                            </div>
                                        </form>
                                    </div>
                                </div>
                                <div class="card shadow-lg rounded">
                                    <div class="card-body text-center">
                                        <input type="text" class="form-control mt-2" placeholder="X" name="X" id="X` + imageIndex + `" value="` + image.X + `">
                                        <input type="text" class="form-control mt-2" placeholder="Y" name="Y" id="Y` + imageIndex + `" value="` + image.Y + `">
                                        <input type="text" class="form-control mt-2" placeholder="Latitude" name="Latitude" id="Latitude` + imageIndex + `" value="` + image.Latitude + `">
                                        <input type="text" class="form-control mt-2" placeholder="Longitude" name="Longitude" id="Longitude` + imageIndex + `" value="` + image.Longitude + `">
                                    </div>
                                </div>
                                <div class="card shadow-lg rounded">
                                    <div class="card-body text-center">
                                        <input type="button" class="btn btn-primary" value="Calculate" id="uploadBtn` + imageIndex + `">
                                        <input type="button" class="btn btn-danger" value="Remove" id="removeBtn` + imageIndex + `">
                                    </div>
                                </div>
                            </div>`
                        );

                        $("#photoDiv" + imageIndex).after(`
                            <div class="col-6 mt-4" id="detectedPhotoDiv` + imageIndex + `">
                                <div class="card shadow-lg rounded">
                                    <div class="card-body">
                                        <div class="row">
                                            <div class="col-12">
                                                <img class="img-fluid" width="20%" height="20%" alt="detected image" src="` + image.detected_path + `" id="img1_"` + imageIndex + `>
                                                <img class="img-fluid" width="20%" height="20%" alt="masked image" src="` + image.output_path + `" id="img2_"` + imageIndex + `>
                                            </div>
                                            <div class="col-12">
                                                <label class="text-primary" id="LAI"><b class="text-success">LAI is</b> ` + image.LAI + `</label>
                                                <br>
                                                <label class="text-primary" id="FVC"><b class="text-success">FVC is</b> ` + image.FVC + `%</label>
                                                <br>
                                                <label class="text-dark" id="metadataLabel"><b>Image metadata</b></label>
                                                <br>
                                                <label class="text-primary" id="Xlabel"><b class="text-secondary">X:</b> ` + image.X + `</label>
                                                <br>
                                                <label class="text-primary" id="Ylabel"><b class="text-secondary">Y:</b> ` + image.Y + `</label>
                                                <br>
                                                <label class="text-primary" id="Height"><b class="text-secondary">Height:</b> ` + image.Latitude + `</label>
                                                <br>
                                                <label class="text-primary" id="Width"><b class="text-secondary">Width:</b> ` + image.Longitude + `</label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `);
                    });
                    if (remaining <= 0){
                        $("#loadMoreBtn").attr('count', 0);
                        $("#loadMoreBtn").html("Get uploaded images (0)")
                        $('#loadMoreBtn').prop('disabled', true);
                        return
                    }

                    $("#loadMoreBtn").attr('count', remaining);
                    $("#loadMoreBtn").html("Get uploaded images (" + remaining + ")")
                }
                else {
                    alert(response.message);
                }
            },
        });
    });

    // var imageLoader = document.getElementById('filePhoto');
    //     imageLoader.addEventListener('change', handleImage, false);

    // function handleImage(e) {
        
    // }

    // Edit for multiple images
    // I didn't try but it should work.
    // Also you need write some CSS code to see all images in container properly.
    // function handleImages(e) {
    //     $('.uploader img').remove();
    //     for(var i = 0; i < e.target.files.length; i++){
    //         var reader = new FileReader();
    //         reader.onload = function (event) {
    //             var $img = $('<img/>');
    //             $img.attr('src', event.target.result);
    //             $('.uploader').append($img);
    //         }
    //         reader.readAsDataURL(e.target.files[i]);
    //     }
    // }
});