$(function(){


	$('#webcam').photobooth().on("image",function( event, dataUrl ){

		$( "#imgShow_WU_FILE" ).attr("src",dataUrl);
		var url = window.document.location.href.toString();
        var u = url.split("?")[1];
        var dataMessage=new Array(6);
        var list=u.split("&");
        for(var i=0;i<4;i++){
            dataMessage[i]=list[i].split("=")[1];
        }
        dataMessage[4]="./static/data/touxiang.jpg";
        dataMessage[5]=dataUrl;
        $( "#loadding" ).removeClass("m_hidden");

             $.ajax({
             type: "post",
             url: "./photograph",
             data:JSON.stringify(dataMessage),
             dataType: "json",
             async:false,
             contentType: 'application/json',
             success: (data) => {

             $( "#loadding" ).addClass("m_hidden");
                    window.location.replace("/result");
             },
             error: (e) => {
             $( "#loadding" ).addClass("m_hidden");
                    window.location.replace("/hello");
             }
         });

	});

	if(!$('#webcam').data('photobooth').isSupported){
		alert('HTML5 webcam is not supported by your browser, please use latest firefox, opera or chrome!');
	}

	$('.photobooth ul li:last').qtip({
		content: {
			text: 'Click here to take pictures',
			title: {
				text: 'Tips',
				button: true
			}
		},
		show: {
			ready: false
		},
		position: {
			target: 'event',
	      	my: 'left center',
	      	at: 'right center'
		},
		style: {
			classes: 'ui-tooltip-shadow ui-tooltip-bootstrap',
			width: 300
		}
	});

	$('#site').qtip({
		content: {
			text: 'Demo from our geek blog: http://www.gbin1.com',
			title: {
				text: 'wlecome',
				button: true
			}
		},
		position: {
			target: 'event',
	      	my: 'bottom center',
	      	at: 'top center',
			viewport: $(window)
		},
		style: {
			classes: 'ui-tooltip-shadow ui-tooltip-jtools'
		}
	});

	$(".imgInput").change(function(){

        var url=URL.createObjectURL($(this)[0].files[0]);
        $( "#imgShow_WU_FILE" ).attr("src",url);

    });

    $(".button").on("click",function(){
            var ext=$("#imgInput").val().split("C:\\fakepath\\")[1].split(".")[1];
            var imgShow_WU_FILE=document.getElementById("imgShow_WU_FILE");
            var imgdatabase64=getImageBase64(imgShow_WU_FILE,ext);
            var imgUrl=$("#text").html();
             var url = window.document.location.href.toString();
            var u = url.split("?")[1];
            var data=[];
            var list=u.split("&");
            for(var i=0;i<4;i++){
                data[i]=list[i].split("=")[1];
            }

            data[4]=imgdatabase64;
            $( "#loadding" ).removeClass("m_hidden");
            setTimeout(function(){
                 $.ajax({
                 type: "post",
                 url: "./filePath",
                 async:false,
                 data:JSON.stringify(data),
                 dataType: "json",

                 contentType: 'application/json',
                 success: (data) => {

                 $( "#loadding" ).addClass("m_hidden");
                        window.location.replace("/result");
                 },
                 error: (e) => {
                 $( "#loadding" ).addClass("m_hidden");
                          window.location.replace("/hello");
                 }
             });
            },1000) ;
    });
});




