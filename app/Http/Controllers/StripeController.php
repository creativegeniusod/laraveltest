<?php
namespace App\Http\Controllers;
use Illuminate\Http\Request;
use App\Helper\Reply;
use Response;
use App\Http\Requests\Stripe\StorePayment;
use Illuminate\Support\Facades\Validator;
use Illuminate\Support\Facades\Input;
use Stripe\Stripe;
use Stripe\Checkout;
use Stripe\Stripe_InvalidRequestError;

class StripeController extends Controller
{
    function payment(Request $request){

        // Setup the validator
        $rules = array('product_name' => 'required',
            'amount' => 'required',
            'product_description' => 'required',
            'email' => 'required|email');

        $validator = Validator::make(Input::all(), $rules);

        // Validate the input and return correct response
        if ($validator->fails()){
          return Response::json(array(
              'success' => false,
              'errors' => $validator->getMessageBag()->toArray()

          ), 400); // 400 being the HTTP code for an invalid request.
        }

        try {
            $amount = $request->amount*100;
            Stripe::setApiKey(env('STRIPE_SECRET', 'sk_test_WtlkuRtoebcVJ6qF1EWQSOJz'));

            $session = \Stripe\Checkout\Session::create([
              'customer_email' => $request->email,
              'payment_method_types' => ['card'],
              'line_items' => [[
                'name' => $request->product_name,
                'description' => $request->product_description,
                'images' => ['https://example.com/t-shirt.png'],
                'amount' => $amount,
                'currency' => 'usd',
                'quantity' => 1,
              ]],
              'success_url' => 'http://laraveltest.stagingwebsites.info/payment/success',
              'cancel_url' => 'http://laraveltest.stagingwebsites.info/payment/cancel',
            ]);

            return Reply::dataOnly(['status' => 'success', 'session_id' => $session->id]);
        } catch (\Stripe\Error\Authentication $e ) {
            return Response::json(array(
                'success' => false,
                'errors' => $e->getMessage()
            ), 500);
        }

    }

    function success(Request $request){
        return view('stripe.success');
    }

    function cancel(Request $request){
        return view('stripe.cancel');
    }
}
