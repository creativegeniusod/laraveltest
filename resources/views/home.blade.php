@extends('layouts.app')

@section('content')
<div class="container">
    <div class="row">
        <div class="col-md-8 col-md-offset-2">
            <div class="panel panel-default">
                <div class="panel-heading">Upload Image</div>

                <div class="panel-body">
                    {!! Form::open(['id'=>'uploadImage','class'=>'ajax-form','method'=>'POST']) !!}
                        <div class="form-content">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="form-group">
                                        <input name="image" type="file" class="custom-file-input form-control" />
                                    </div>
                                </div>
                            </div>
                            <button id="save-form" type="button" class="btnSubmit">Upload Image</button>
                            <a style="display:none" id="download_pdf" href="#" class="btnSubmit" target="_blank" > Download PDF </a>
                        </div>
                    {!! Form::close() !!}
                </div>
            </div>
        </div>
    </div>
</div>
@endsection

@push('footer-script')
  <script>
      $('#save-form').click(function () {
        $(".serverResponce").remove();
          $.easyAjax({
              url: '{{route('media.upload_image')}}',
              type: "POST",
              container: '#uploadImage',
              redirect: false,
              file:true,
              data: $('#uploadImage').serialize(),
              success: function (response) {
                  console.log(response);
                  if(response.status == 'success'){
                      console.log(response);
                      $('#download_pdf').show();
                      $('#download_pdf').attr("href", response.pdf_file); // Set herf value
                  }else{                     
                    $('input[type=file]').after('<div class="help-block serverResponce">'+response.msg+'</div>');                    
                  }
              }
          })
      });
  </script>
@endpush
