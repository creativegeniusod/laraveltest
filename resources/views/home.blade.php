@extends('layouts.app')

@section('content')
<div class="container">
    <div class="row">
        <div class="col-md-8 col-md-offset-2">
            <div class="panel panel-default">
                <div class="panel-heading">Stripe Payment Form</div>

                <div class="panel-body">
                    {!! Form::open(['id'=>'stripeForm','class'=>'ajax-form','method'=>'POST']) !!}
                        <div class="form-content">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="form-group">
                                        <label>Product Name</label>
                                        <input name="product_name" type="text" class="form-control" />
                                    </div>
                                    <div class="form-group">
                                        <label>Product Description</label>
                                        <input name="product_description" type="text" class="form-control" />
                                    </div>
                                    <div class="form-group">
                                        <label>Amount</label>
                                        <input name="amount" type="text" class="form-control" />
                                    </div>
                                    <div class="form-group">
                                        <label>Customer Email</label>
                                        <input name="email" type="text" class="form-control" />
                                    </div>
                                </div>
                            </div>
                            <button id="save-form" type="button" class="btnSubmit">Submit</button>
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
          $.easyAjax({
              url: '{{route('stripe.payment')}}',
              type: "POST",
              container: '#stripeForm',
              redirect: false,
              data: $('#stripeForm').serialize(),
              success: function (response) {
                  console.log(response);
                  if(response.status == 'success'){
                      console.log(response);
                      redirectStripe(response.session_id);
                  }
              }
          })
      });


      function redirectStripe(checkout_session_id) {
          var stripe = Stripe('pk_test_ugWkHgHhBw5cHexAkfTkLlUp');
          stripe.redirectToCheckout({
            // Make the id field from the Checkout Session creation API response
            // available to this file, so you can provide it as parameter here
            sessionId: checkout_session_id
          }).then(function (result) {
            // If `redirectToCheckout` fails due to a browser or network
            // error, display the localized error message to your customer
            // using `result.error.message`.
          });
      }

  </script>
@endpush
