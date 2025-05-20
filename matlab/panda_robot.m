
function robot = panda_robot()

clear L

%[theta d a alpha]

mm = 1e-3;

% robot length values (metres)
d = [333 0 316 0 384 0 107]'*mm;

a = [0 0 82.5 -82.5 0 88 0]'*mm;

alpha = [-pi/2 pi/2 pi/2 -pi/2 pi/2 pi/2 0]';

L(1) = Revolute('d', d(1), 'a', a(1), 'alpha', alpha(1));
L(2) = Revolute('d', d(2), 'a', a(2), 'alpha', alpha(2));
L(3) = Revolute('d', d(3), 'a', a(3), 'alpha', alpha(3));
L(4) = Revolute('d', d(4), 'a', a(4), 'alpha', alpha(4));
L(5) = Revolute('d', d(5), 'a', a(5), 'alpha', alpha(5));
L(6) = Revolute('d', d(6), 'a', a(6), 'alpha', alpha(6));
L(7) = Revolute('d', d(7), 'a', a(7), 'alpha', alpha(7));

L(1).m = 1;
L(2).m = 0.;
L(3).m = 3;
L(4).m = 0.;
L(5).m = 5.;
L(6).m = 0.;
L(7).m = 2.5;

L(1).I = [ 0.1, 0.1, 0.1, 0., 0., 0. ];
L(2).I = [ 0.1, 0.1, 0.1, 0., 0., 0. ];
L(3).I = [ 0.1, 0.1, 0.1, 0., 0., 0. ];
L(4).I = [ 0.1, 0.1, 0.1, 0., 0., 0. ];
L(5).I = [ 0.1, 0.1, 0.1, 0., 0., 0. ];
L(6).I = [ 0.1, 0.1, 0.1, 0., 0., 0. ];
L(7).I = [ 0.1, 0.1, 0.1, 0., 0., 0. ];

L(1).r = [ -d(1)/2 0.0 0.0];
L(2).r = [ 0 0 d(3)/2];
L(3).r = [ 0 0.0 0.0];
L(4).r = [ 0.0 0.0 0.0];
L(5).r = [ 0 -d(5)/2 0.0];
L(6).r = [ 0.0 0.0 0.0];
L(7).r = [ 0.0 0.0 0.0];

L(1).Jm = 0;
L(2).Jm = 0;
L(3).Jm = 0;
L(4).Jm = 0;
L(5).Jm = 0;
L(6).Jm = 0;
L(7).Jm = 0;

robot = SerialLink(L,  'name', 'panda');

clear L
